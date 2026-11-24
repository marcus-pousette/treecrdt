/**
 * A view groups nodes by keys that an app derives from each node's own payload, and gives each
 * group an order that replicates.
 *
 * - Membership is derived on each device from materialized state and never written as ops, so it
 *   can be redefined (bump `version`) without rewriting history, and works on decrypted payloads.
 * - Order is sparse. A member someone placed has one record node, whose order key sits strictly
 *   between the effective keys of the neighbours shown. Every other member sorts by a default key
 *   built from `rank`, in the same key space. Only a placement writes, so nothing automatic can
 *   overwrite one. Concurrent placements of one member resolve last-writer-wins, and placements of
 *   different members merge by key.
 * - Records live in the same document, in a subtree that the first placement adds under the root
 *   and every replica then has. Code that walks the tree or listens to `onMaterialized` must skip
 *   its nodes, whose ids all start with `VIEW_NODE_PREFIX`.
 * - A record never creates membership. It is ignored while its member is out of the group, and
 *   applies again when the member returns.
 */
import { hmac } from '@noble/hashes/hmac';
import { sha256 } from '@noble/hashes/sha2';
import type {
  DurableMaterializationSubscription,
  LocalWriteOptions,
  MaterializationEvent,
  TreecrdtEngine,
} from './engine.js';
import { bytesToHex, hexToBytes, ROOT_NODE_ID_HEX } from './ids.js';
import type { Operation, ReplicaId } from './index.js';
import type { SqliteRunner } from './sqlite.js';

/**
 * First 6 id bytes of every node a view writes, so a consumer can skip them. Shorter would risk
 * matching random node ids. The other 10 bytes are a keyed hash.
 */
export const VIEW_NODE_PREFIX = 'e7c5e1f5a0d1';
const TABLE = 'treecrdt_view_members';
const MAX_RANK = 2 ** 48 - 1;

export type TreecrdtViewClient = Pick<
  TreecrdtEngine,
  'docId' | 'local' | 'tree' | 'onMaterializedDurable'
> & { runner: SqliteRunner };

export type TreecrdtViewNode = { id: string; payload: Uint8Array };

export type TreecrdtViewOptions = {
  name: string;
  /** Bump when `keys`, `rank` or `data` change meaning; the derived index is rebuilt. */
  version: number;
  /** Group keys of a live node with a payload. Runs in the app, so it may decrypt. */
  keys: (node: TreecrdtViewNode) => string[];
  /**
   * Default position of a member no record placed, such as its creation time. Clamped into
   * [0, 2^48); a non-finite rank counts as 0.
   */
  rank: (node: TreecrdtViewNode) => number;
  /**
   * Small JSON value kept with each membership and returned by `members`. `onChange` does not
   * report a change of it alone, which may come with every edit.
   */
  data?: (node: TreecrdtViewNode) => unknown;
  /** Replica that writes order records. */
  replica: ReplicaId;
  /**
   * Keys the ids of group and record nodes, together with the doc id. Defaults to the doc id
   * alone. Under end-to-end encryption, pass a secret the document's members share, so a relay
   * cannot test guesses of group keys.
   */
  secret?: Uint8Array;
  /** Options for each order write, such as a write id or an auth session. */
  writeOptions?: () => LocalWriteOptions;
  /**
   * Receives what the view skipped: a node its callbacks threw for, a failed index update (retried
   * with the next event), or a failing `onChange` listener. A node the callbacks threw for is in no
   * group until it changes again or `version` does: bump `version` once they can map it, such as
   * when a key arrives that decrypts it.
   */
  onError?: (error: unknown, node?: string) => void;
};

export type TreecrdtViewMember = {
  id: string;
  /** Whether a record placed it; false at its default position. */
  placed: boolean;
  /** Effective order key, hex. */
  key: string;
  /** What `options.data` returned for the member, if given. */
  data?: unknown;
};

export type TreecrdtView = {
  /** Members of each requested group in display order. Groups without members are absent. */
  members: (groups: readonly string[]) => Promise<Map<string, TreecrdtViewMember[]>>;
  /**
   * Shows `member` of `group` right before `before` (null: last), in the order this replica shows
   * now, and returns the ops written: the record, plus the view's root and group nodes the first
   * time. Writes nothing if the member is already there or no longer in the group. Places it last
   * if `before` is no longer in the group. Calls run one at a time, each against the order the
   * previous one left.
   */
  place: (group: string, member: string, before: string | null) => Promise<Operation[]>;
  /**
   * Called once the index has applied each event, with the groups whose members or order changed
   * (possibly none) and the event itself.
   */
  onChange: (listener: (groups: string[], event: MaterializationEvent) => void) => () => void;
  /** Keeps the index up to date: `settled()` waits for everything committed, `close()` stops it. */
  subscription: DurableMaterializationSubscription;
};

const encoder = new TextEncoder();
const hex4 = (digit: number) => digit.toString(16).padStart(4, '0');

const clampRank = (rank: number) =>
  Number.isFinite(rank) ? Math.min(MAX_RANK, Math.max(0, Math.floor(rank))) : 0;

/** The first three order-key digits of a rank: a prefix of every default key with that rank. */
const rankDigits = (rank: number): string => {
  const hi = Math.floor(rank / 2 ** 32);
  const lo = rank % 2 ** 32;
  return hex4(hi & 0xffff) + hex4((lo >>> 16) & 0xffff) + hex4(lo & 0xffff);
};

/** The rank that a key's first three digits encode. */
const keyRank = (key: string): number => {
  const digits = key.padEnd(12, '0');
  return (
    parseInt(digits.slice(0, 4), 16) * 2 ** 32 +
    parseInt(digits.slice(4, 8), 16) * 2 ** 16 +
    parseInt(digits.slice(8, 12), 16)
  );
};

/**
 * Opens the view `options.name` over `client` and keeps its index up to date until its subscription
 * closes. The index lives in the client's database, so reopening the view resumes where it stopped.
 */
export async function createTreecrdtView(
  client: TreecrdtViewClient,
  options: TreecrdtViewOptions,
): Promise<TreecrdtView> {
  if (!client.onMaterializedDurable) {
    throw new Error('createTreecrdtView needs onMaterializedDurable');
  }
  const { runner } = client;
  const { name } = options;
  const report = (error: unknown, node?: string) => {
    if (options.onError) options.onError(error, node);
    else console.error(`TreeCRDT view ${name}:`, error, node ?? '');
  };
  // Synchronous HMAC: indexing hashes once per membership, and async crypto would add a round trip
  // per row. The doc id is always in the message, so documents that share a secret get different
  // ids.
  const secret = options.secret ?? encoder.encode(client.docId);
  const viewId = (...parts: string[]) => {
    const mac = hmac(sha256, secret, encoder.encode([client.docId, name, ...parts].join('\u0000')));
    return VIEW_NODE_PREFIX + bytesToHex(mac.subarray(0, 10));
  };
  const root = viewId('root');
  const groupNodes = new Map<string, string>();
  const groupNode = (group: string) => {
    let id = groupNodes.get(group);
    if (!id) {
      id = viewId('group', group);
      groupNodes.set(group, id);
    }
    return id;
  };
  const recordNode = (group: string, member: string) => viewId('record', group, member);
  /**
   * Rank digits, then the member id, so equal ranks sort by id, then a final 1 digit, so a key can
   * always be placed below it.
   */
  const defaultKey = (rank: number, member: string) => rankDigits(rank) + member + '0001';

  // One row per member of a group. `default_key` orders a member no record placed, and
  // `group_node` and `record` name the nodes that place it.
  await runner.exec(`
    CREATE TABLE IF NOT EXISTS ${TABLE} (
      view TEXT NOT NULL,
      node BLOB NOT NULL,
      group_key TEXT NOT NULL,
      default_key BLOB NOT NULL,
      data TEXT,
      group_node BLOB NOT NULL,
      record BLOB NOT NULL,
      PRIMARY KEY (view, node, group_key)
    ) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS ${TABLE}_group ON ${TABLE} (view, group_key);
    CREATE INDEX IF NOT EXISTS ${TABLE}_record ON ${TABLE} (record);
  `);

  type Row = { node: string; group: string; defaultKey: string; data: string | null };

  /**
   * Applies one event and returns the groups a member joined, left or moved in. A change only
   * names the node: its stored payload and tombstone are read again, since a coalesced change can
   * carry a payload other than the stored one.
   */
  const apply = async (event: MaterializationEvent): Promise<string[]> => {
    const nodes = new Set<string>();
    const records = new Set<string>();
    for (const change of event.changes) {
      if (!change.node.startsWith(VIEW_NODE_PREFIX)) nodes.add(change.node);
      else if (change.kind !== 'payload') records.add(change.node);
    }
    const affected = new Set<string>();
    if (nodes.size > 0) {
      const named = JSON.stringify([...nodes]);
      const stored = JSON.parse(
        (await runner.getText(
          `SELECT json_group_array(json_array(lower(hex(n.node)), n.tombstone, lower(hex(p.payload))))
           FROM tree_nodes n LEFT JOIN tree_payload p ON p.node = n.node
           WHERE n.node IN (SELECT unhex(value) FROM json_each(?1))`,
          [named],
        )) ?? '[]',
      ) as [string, number, string][];
      const payloads = new Map<string, Uint8Array>();
      for (const [node, tombstone, payload] of stored) {
        if (tombstone === 0 && payload) payloads.set(node, hexToBytes(payload));
      }
      const before = JSON.parse(
        (await runner.getText(
          `SELECT json_group_array(json_array(lower(hex(node)), group_key, lower(hex(default_key)), data))
           FROM ${TABLE} WHERE view = ?1 AND node IN (SELECT unhex(value) FROM json_each(?2))`,
          [name, named],
        )) ?? '[]',
      ) as [string, string, string, string | null][];
      const old = new Map<string, Row>();
      for (const [node, group, defaultKey, data] of before) {
        old.set(`${node}\u0000${group}`, { node, group, defaultKey, data });
      }
      const next = new Map<string, Row>();
      for (const node of nodes) {
        const payload = payloads.get(node);
        if (!payload) continue;
        try {
          const groups = new Set(options.keys({ id: node, payload }));
          if (groups.size === 0) continue;
          const key = defaultKey(clampRank(options.rank({ id: node, payload })), node);
          const data = options.data
            ? JSON.stringify(options.data({ id: node, payload }) ?? null)
            : null;
          for (const group of groups) {
            next.set(`${node}\u0000${group}`, { node, group, defaultKey: key, data });
          }
        } catch (error) {
          // A payload this device cannot map joins no group until the node or the version changes.
          report(error, node);
        }
      }
      const gone = [...old].filter(([id]) => !next.has(id)).map(([, row]) => row);
      const changed = [...next].filter(([id, row]) => {
        const was = old.get(id);
        return !was || was.defaultKey !== row.defaultKey || was.data !== row.data;
      });
      for (const row of gone) affected.add(row.group);
      // Report a group when a member joins it or moves in it, not when only its data changes, which
      // may happen on every edit.
      for (const [id, row] of changed) {
        if (old.get(id)?.defaultKey !== row.defaultKey) affected.add(row.group);
      }
      // Write before deleting, so a concurrent read never misses a member that stays.
      if (changed.length > 0) {
        const rows = changed.map(([, { node, group, defaultKey, data }]) => [
          node,
          group,
          defaultKey,
          data,
          groupNode(group),
          recordNode(group, node),
        ]);
        await runner.run(
          `INSERT OR REPLACE INTO ${TABLE} (view, node, group_key, default_key, data, group_node, record)
           SELECT ?1, unhex(value->>0), value->>1, unhex(value->>2), value->>3, unhex(value->>4),
                  unhex(value->>5)
           FROM json_each(?2)`,
          [name, JSON.stringify(rows)],
        );
      }
      if (gone.length > 0) {
        await runner.run(
          `DELETE FROM ${TABLE} WHERE view = ?1
           AND (node, group_key) IN (SELECT unhex(value->>0), value->>1 FROM json_each(?2))`,
          [name, JSON.stringify(gone.map((row) => [row.node, row.group]))],
        );
      }
    }
    if (records.size > 0) {
      // A record only matters while its member is in the group it was placed in.
      const groups = JSON.parse(
        (await runner.getText(
          `SELECT json_group_array(DISTINCT group_key) FROM ${TABLE}
           WHERE view = ?1 AND record IN (SELECT unhex(value) FROM json_each(?2))`,
          [name, JSON.stringify([...records])],
        )) ?? '[]',
      ) as string[];
      for (const group of groups) affected.add(group);
    }
    return [...affected];
  };

  const listeners = new Set<(groups: string[], event: MaterializationEvent) => void>();
  const subscription = await client.onMaterializedDurable(
    `treecrdt-view:${name}`,
    async (event) => {
      const groups = await apply(event);
      for (const listener of listeners) {
        try {
          listener(groups, event);
        } catch (error) {
          report(error);
        }
      }
    },
    {
      version: options.version,
      reset: async () => {
        await runner.run(`DELETE FROM ${TABLE} WHERE view = ?1`, [name]);
      },
      onError: (error) => report(error),
    },
  );

  const members = async (groups: readonly string[]) => {
    const out = new Map<string, TreecrdtViewMember[]>();
    if (groups.length === 0) return out;
    // A record places its member only while it is live under the member's group node.
    const text = await runner.getText(
      `SELECT json_group_array(json_array(group_key, node, key, placed, data)) FROM (
         SELECT m.group_key AS group_key, lower(hex(m.node)) AS node, m.data AS data,
                lower(hex(COALESCE(r.order_key, m.default_key))) AS key,
                r.order_key IS NOT NULL AS placed
         FROM ${TABLE} m
         JOIN tree_nodes t ON t.node = m.node AND t.tombstone = 0
         LEFT JOIN tree_nodes r ON r.node = m.record AND r.parent = m.group_node AND r.tombstone = 0
         WHERE m.view = ?1 AND m.group_key IN (SELECT value FROM json_each(?2))
         ORDER BY m.group_key, COALESCE(r.order_key, m.default_key), m.node
       )`,
      [name, JSON.stringify(groups)],
    );
    const rows = JSON.parse(text ?? '[]') as [string, string, string, number, string | null][];
    for (const [group, id, key, placed, data] of rows) {
      const list = out.get(group) ?? [];
      list.push({
        id,
        key,
        placed: placed === 1,
        ...(data === null ? {} : { data: JSON.parse(data) as unknown }),
      });
      out.set(group, list);
    }
    return out;
  };

  const writer = client.local.forReplica(options.replica);
  const writeOptions = () => options.writeOptions?.();

  const placeNow = async (group: string, member: string, before: string | null) => {
    if (before === member) return [];
    // Place against the order the index shows once it has applied everything committed.
    await subscription.settled();
    const order = (await members([group])).get(group) ?? [];
    const at = order.findIndex((m) => m.id === member);
    // The member left the group meanwhile, such as by a concurrent edit: nothing to place.
    if (at < 0) return [];
    const rest = order.filter((m) => m.id !== member);
    let index = before === null ? rest.length : rest.findIndex((m) => m.id === before);
    // The member to drop before left the group meanwhile: place the member last.
    if (index < 0) index = rest.length;
    // Already shown there: nothing to write.
    if (order[at - 1]?.id === rest[index - 1]?.id && order[at + 1]?.id === rest[index]?.id) {
      return [];
    }
    const leftKey = rest[index - 1]?.key ?? null;
    let rightKey = rest[index]?.key ?? null;
    if (leftKey !== null && rightKey === null) {
      // Dropped last: stay below every default key of a later rank, so members that join later
      // still list after this one.
      const bound = rankDigits(clampRank(keyRank(leftKey) + 1));
      if (bound > leftKey) rightKey = bound;
    }
    const left = leftKey === null ? null : hexToBytes(leftKey);
    const right = rightKey === null ? null : hexToBytes(rightKey);
    const [grp, rec] = [groupNode(group), recordNode(group, member)];
    // Not tree.read, which would also list every top-level node and every record of the group.
    const [hasRoot, hasGroup, hasRecord] = await Promise.all(
      [root, grp, rec].map(async (id) => (await client.tree.get(id)) !== undefined),
    );
    const ops: Operation[] = [];
    // Ids are deterministic, so replicas that create these nodes concurrently converge: an insert
    // of an existing node is a move. They carry no intent of their own, so that is harmless. Open
    // bounds look up no sibling, unlike 'last', which fails when a peer's insert of the same node
    // lands after the check above.
    const anywhere = { type: 'between' as const, left: null, right: null };
    if (!hasRoot) {
      ops.push(await writer.insert(ROOT_NODE_ID_HEX, root, anywhere, null, writeOptions()));
    }
    if (!hasGroup) ops.push(await writer.insert(root, grp, anywhere, null, writeOptions()));
    const placement = { type: 'between' as const, left, right };
    ops.push(
      hasRecord
        ? await writer.move(rec, grp, placement, writeOptions())
        : await writer.insert(grp, rec, placement, null, writeOptions()),
    );
    return ops;
  };

  // One placement at a time: a second drop is placed against the order after the first, which is
  // the order the user saw when making it.
  let placing: Promise<unknown> = Promise.resolve();
  const place = (group: string, member: string, before: string | null) => {
    const next = placing.then(() => placeNow(group, member, before));
    placing = next.catch(() => undefined);
    return next;
  };

  return {
    members,
    place,
    onChange: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    subscription,
  };
}
