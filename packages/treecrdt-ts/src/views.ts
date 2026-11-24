/**
 * A view groups nodes by keys that an app derives from each node's own payload, and orders each
 * group as a subtree of the document, so the order replicates and can hold folders and nesting.
 *
 * - Membership is derived on each device from materialized state and never written as ops, so it
 *   can be redefined (bump `version`) without rewriting history, and works on decrypted payloads.
 *   It reads each node's own tombstone, so a member under a deleted node leaves its groups only
 *   where the engine also marks the deleted node's descendants and reports them as changed.
 * - A group nobody ordered lists its members by a default key built from `rank`, and stores
 *   nothing. The first `place` or `createFolder` orders it: it adds the group's node under the
 *   view root, and a reference node per member at that member's default key, so the order shown
 *   stays the same. From then on the order is a depth-first walk of the group's subtree:
 *   references and folders, ordered as ordinary siblings, and whatever is nested under either.
 * - Automatic writes are create-only inserts (`insertIfAbsent`) at keys every device derives
 *   alike, so devices that order a group or add a reference at once agree, and an automatic write
 *   never moves a reference someone placed. A drag is a plain move.
 * - A reference never creates membership. One whose member left the group is skipped, its
 *   children shown in its place, and applies again when the member returns. A member without a
 *   reference lists among the top-level items at its default key, where `repair` puts one.
 * - View nodes live in the same document, under a view root that the first ordering adds among the
 *   root's children and every replica then has. Everything that reads the root's children, walks
 *   the tree or listens to `onMaterialized` must skip ids that start with `VIEW_NODE_PREFIX`, and a
 *   writer whose capability covers only a subtree cannot order a group.
 * - Views read the SQLite extension's `tree_nodes` and `tree_payload` tables directly, for order
 *   keys, deleted nodes and nodes not yet in the tree, which `tree.read` leaves out, so they run on
 *   the SQLite engines only and follow that schema.
 */
import { hmac } from '@noble/hashes/hmac';
import { sha256 } from '@noble/hashes/sha2';
import { randomBytes } from '@noble/hashes/utils';
import type {
  DurableMaterializationSubscription,
  LocalWriteOptions,
  MaterializationEvent,
  TreecrdtEngine,
} from './engine.js';
import { bytesToHex, hexToBytes, ROOT_NODE_ID_HEX } from './ids.js';
import type { Operation, ReplicaId } from './index.js';
import type { SqliteRunner, TreecrdtSqlitePlacement } from './sqlite.js';

/**
 * First 6 id bytes of every node a view writes, so a consumer can skip them. Shorter would risk
 * matching random node ids. The other 10 bytes are a keyed hash, or random for a folder. The low
 * bit of the byte after the prefix is set in a folder's id and clear in every other view node's,
 * so a replica that never indexed a reference's member can still tell the reference from a folder.
 */
export const VIEW_NODE_PREFIX = 'e7c5e1f5a0d1';
const MEMBERS = 'treecrdt_view_members';
const GROUPS = 'treecrdt_view_groups';
const MAX_RANK = 2 ** 48 - 1;
/** Marks a folder id: see `VIEW_NODE_PREFIX`. */
const FOLDER_BIT = 0x01;

export type TreecrdtViewClient = Pick<
  TreecrdtEngine,
  'docId' | 'local' | 'onMaterializedDurable'
> & {
  runner: SqliteRunner;
};

export type TreecrdtViewNode = { id: string; payload: Uint8Array };

export type TreecrdtViewOptions = {
  name: string;
  /**
   * Bump when `keys`, `rank`, `data`, `reference` or `secret` change; the index is rebuilt. A new
   * `reference` gives every reference a new id, and a new `secret` every view node, so groups lose
   * their order: members list at their default keys again, and the old view nodes stay in the
   * document. Choose both before any group is ordered.
   */
  version: number;
  /** Group keys of a live node with a payload. Runs in the app, so it may decrypt. */
  keys: (node: TreecrdtViewNode) => string[];
  /**
   * A member's default position, such as its creation time: where it lists while it has no
   * reference, and where `repair` writes one. Clamped into [0, 2^48); a non-finite rank counts as
   * 0.
   */
  rank: (node: TreecrdtViewNode) => number;
  /**
   * Small JSON value kept with each membership and returned with the member. `onChange` does not
   * report a change of it alone, which may come with every edit.
   */
  data?: (node: TreecrdtViewNode) => unknown;
  /**
   * What a reference stands for. 'member' (default): the member. 'parent': the member in its
   * context, its parent, so members of a group under one parent share a reference, and a member
   * deleted and created again under the same parent gets its place back. A move re-derives it.
   */
  reference?: 'member' | 'parent';
  /** Replica that writes the view's ops. */
  replica: ReplicaId;
  /**
   * Keys the ids of the view's root, group and reference nodes, together with the doc id.
   * Defaults to the doc id alone. Under end-to-end encryption, pass a secret the document's
   * members share, so a relay cannot test guesses of group keys. Changing it later drops every
   * group's order (see `version`).
   */
  secret?: Uint8Array;
  /** Options for each write, such as a write id or an auth session. */
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
  /** Whether its reference is in the group's subtree; false where it lists at its default key. */
  placed: boolean;
  /** What `options.data` returned for the member, if given. */
  data?: unknown;
};

/**
 * An item of a group as `tree` returns it. Members that share a reference (`reference: 'parent'`)
 * list one after another, with the reference's children under the first.
 */
export type TreecrdtViewItem =
  | { kind: 'member'; id: string; data?: unknown; children: TreecrdtViewItem[] }
  | { kind: 'folder'; id: string; payload: Uint8Array | null; children: TreecrdtViewItem[] };

/** Where an item goes, in the order shown. */
export type TreecrdtViewPosition = {
  /** The member or folder to go into. Null or omitted: the top level. */
  parent?: string | null;
  /** The item of `parent` to go right before. Null or omitted: last. */
  before?: string | null;
};

export type TreecrdtView = {
  /** Members of each requested group in display order. Groups without members are absent. */
  members: (groups: readonly string[]) => Promise<Map<string, TreecrdtViewMember[]>>;
  /** Members and folders of `group`, nested as in its subtree, in display order. */
  tree: (group: string) => Promise<TreecrdtViewItem[]>;
  /**
   * Moves `item`, a member or folder of `group`, to `position` in the order this replica shows now,
   * ordering the group first if needed, and returns the ops written. Writes nothing if the item is
   * already there or no longer in the group. Goes to the top level if `parent` is no longer in the
   * group, and last if `before` is no longer in `parent`.
   */
  place: (group: string, item: string, position?: TreecrdtViewPosition) => Promise<Operation[]>;
  /**
   * Writes a reference for each member of an ordered group that has none, at the member's default
   * key at the top level, and returns the ops written. Call it after the app's own writes and once
   * sync settles. Repairs `groups`, or every ordered group.
   */
  repair: (groups?: readonly string[]) => Promise<Operation[]>;
  /** Adds a folder at `position`, ordering the group first if needed. */
  createFolder: (
    group: string,
    options?: TreecrdtViewPosition & { payload?: Uint8Array | null },
  ) => Promise<{ id: string; ops: Operation[] }>;
  /** Moves the folder's items to its place, in order, then deletes it. */
  removeFolder: (group: string, folder: string) => Promise<Operation[]>;
  /**
   * Called once the index has applied each event, with the groups whose members or order changed
   * (possibly none) and the event itself. A change under a group's node maps to the group once
   * this device has indexed a member of it or written to it.
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

/** A bound above `key` and below every default key of a later rank, if there is one. */
const rankBound = (key: string): string | null => {
  const bound = rankDigits(clampRank(keyRank(key) + 1));
  return bound > key ? bound : null;
};

const isFolder = (node: string) => {
  const marker = parseInt(node.slice(VIEW_NODE_PREFIX.length, VIEW_NODE_PREFIX.length + 2), 16);
  return node.startsWith(VIEW_NODE_PREFIX) && (marker & FOLDER_BIT) !== 0;
};

const between = (left: string | null, right: string | null): TreecrdtSqlitePlacement => ({
  type: 'between',
  left: left === null ? null : hexToBytes(left),
  right: right === null ? null : hexToBytes(right),
});
const at = (key: string): TreecrdtSqlitePlacement => ({ type: 'at', key: hexToBytes(key) });
/** Open bounds look up no sibling, unlike 'last'. */
const anywhere = between(null, null);

/** A member's row in the index. */
type Membership = {
  node: string;
  group: string;
  defaultKey: string;
  data: string | null;
  reference: string;
};

/** A node of an ordered group's subtree, or a reference that the walk does not reach. */
type Slot = {
  node: string;
  /** Order key, hex. A reference the walk does not reach lists at its default key. */
  key: string;
  folder: boolean;
  /** A folder's payload. */
  payload: Uint8Array | null;
  /** The members a reference stands for, by default key. None: its children show in its place. */
  members: Membership[];
  /**
   * A reference the walk does not reach: none yet, one deleted or under a deleted node, or one
   * under a node whose insert has not arrived.
   */
  pending: boolean;
  /** Null at the top level. */
  parent: Slot | null;
  children: Slot[];
};

type Layout = {
  group: string;
  groupNode: string;
  /** Whether the group node is in the tree. */
  ordered: boolean;
  top: Slot[];
  /** Slots of the subtree's nodes. */
  nodes: Map<string, Slot>;
  /** The slot of each member's reference. */
  references: Map<string, Slot>;
};

const compareHex = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/** Siblings sort as TreeCRDT orders children: by order key, then node id. */
const slotOrder = (a: Slot, b: Slot) => compareHex(a.key, b.key) || compareHex(a.node, b.node);

/** Whether a slot shows as an item: a folder, or a reference with members. */
const shows = (slot: Slot) => slot.folder || slot.members.length > 0;

/** The nearest ancestor that shows as an item. */
const shownParent = (slot: Slot): Slot | null => {
  let parent = slot.parent;
  while (parent && !shows(parent)) parent = parent.parent;
  return parent;
};

/** The first item `slot` shows: itself, or for a reference whose member left, its children's. */
const firstShown = (slot: Slot): Slot | null => {
  if (shows(slot)) return slot;
  for (const child of slot.children) {
    const shown = firstShown(child);
    if (shown) return shown;
  }
  return null;
};

/** The item shown right after `slot` under its shown parent, null for the last. */
const nextShown = (top: Slot[], slot: Slot): Slot | null => {
  for (let current = slot; ; ) {
    const siblings = current.parent ? current.parent.children : top;
    for (const sibling of siblings.slice(siblings.indexOf(current) + 1)) {
      const shown = firstShown(sibling);
      if (shown) return shown;
    }
    // The children of a reference whose member left show in its place, so its next items follow.
    if (!current.parent || shows(current.parent)) return null;
    current = current.parent;
  }
};

/**
 * The placement under `container` (null: the top level) right after `left` and before `right`,
 * null for an end.
 */
const placementBetween = (
  container: Slot | null,
  left: { key: string } | null,
  right: { key: string } | null,
) =>
  between(
    left?.key ?? null,
    // Last at the top level: stay below every default key of a later rank, so members that join
    // later still list after this one.
    right ? right.key : !container && left ? rankBound(left.key) : null,
  );

const parseData = (data: string | null) =>
  data === null ? {} : { data: JSON.parse(data) as unknown };

/** SQL for whether `tree_nodes` row `t` is in the tree, not a placeholder: see `index`. */
const inTreeSql = (t: string) => `(${t}.parent IS NOT NULL OR ${t}.node = X'${ROOT_NODE_ID_HEX}')`;

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
  const byParent = options.reference === 'parent';
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
    const id = mac.subarray(0, 10);
    id[0] &= ~FOLDER_BIT;
    return VIEW_NODE_PREFIX + bytesToHex(id);
  };
  const folderId = () => {
    const id = randomBytes(10);
    id[0] |= FOLDER_BIT;
    return VIEW_NODE_PREFIX + bytesToHex(id);
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
  /**
   * Rank digits, then what the reference stands for, so equal ranks sort by id and no two
   * references share a default key, then a final 1 digit, so a key can always be placed below it.
   */
  const defaultKey = (rank: number, target: string) => rankDigits(rank) + target + '0001';

  // One row per member of a group. `default_key` orders it while its reference is not in the
  // group's subtree, and `reference` names that node. The groups table maps a group node back to
  // its key, for changes under it.
  await runner.exec(`
    CREATE TABLE IF NOT EXISTS ${MEMBERS} (
      view TEXT NOT NULL,
      node BLOB NOT NULL,
      group_key TEXT NOT NULL,
      default_key BLOB NOT NULL,
      data TEXT,
      reference BLOB NOT NULL,
      PRIMARY KEY (view, node, group_key)
    ) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS ${MEMBERS}_group ON ${MEMBERS} (view, group_key);
    CREATE TABLE IF NOT EXISTS ${GROUPS} (
      group_node BLOB PRIMARY KEY,
      view TEXT NOT NULL,
      group_key TEXT NOT NULL
    ) WITHOUT ROWID;
  `);

  /** Groups this session has recorded in the groups table. */
  const recorded = new Set<string>();
  const recordGroups = async (groups: Iterable<string>) => {
    const fresh = [...new Set(groups)].filter((group) => !recorded.has(group));
    if (fresh.length === 0) return;
    await runner.run(
      `INSERT OR IGNORE INTO ${GROUPS} (group_node, view, group_key)
       SELECT unhex(value->>0), ?1, value->>1 FROM json_each(?2)`,
      [name, JSON.stringify(fresh.map((group) => [groupNode(group), group]))],
    );
    for (const group of fresh) recorded.add(group);
  };

  /**
   * Indexes `nodes` again and returns the groups a member joined, left or moved in. A change only
   * names the node: its stored payload, tombstone and parent are read again, since a coalesced
   * change can carry a payload other than the stored one.
   */
  const index = async (nodes: Set<string>): Promise<Set<string>> => {
    const named = JSON.stringify([...nodes]);
    const stored = JSON.parse(
      (await runner.getText(
        `SELECT json_group_array(json_array(lower(hex(n.node)), n.tombstone, lower(hex(p.payload)),
                                            lower(hex(n.parent))))
         FROM tree_nodes n LEFT JOIN tree_payload p ON p.node = n.node
         WHERE n.node IN (SELECT unhex(value) FROM json_each(?1))`,
        [named],
      )) ?? '[]',
    ) as [string, number, string, string][];
    const live = new Map<string, { payload: Uint8Array; parent: string }>();
    for (const [node, tombstone, payload, parent] of stored) {
      // An op that names a node before its insert arrives, such as an edit, leaves a placeholder
      // row with no parent (hex reads it as ''), which is not in the tree yet. Only the root is in
      // it without one.
      const inTree = parent !== '' || node === ROOT_NODE_ID_HEX;
      if (tombstone === 0 && payload && inTree) {
        live.set(node, { payload: hexToBytes(payload), parent });
      }
    }
    const before = JSON.parse(
      (await runner.getText(
        `SELECT json_group_array(json_array(lower(hex(node)), group_key, lower(hex(default_key)), data,
                                            lower(hex(reference))))
         FROM ${MEMBERS} WHERE view = ?1 AND node IN (SELECT unhex(value) FROM json_each(?2))`,
        [name, named],
      )) ?? '[]',
    ) as [string, string, string, string | null, string][];
    const old = new Map<string, Membership>();
    for (const [node, group, defaultKey, data, reference] of before) {
      old.set(`${node}\u0000${group}`, { node, group, defaultKey, data, reference });
    }
    const next = new Map<string, Membership>();
    for (const node of nodes) {
      const state = live.get(node);
      // A member in its context needs a parent, which only a root lacks.
      if (!state || (byParent && !state.parent)) continue;
      const { payload } = state;
      const target = byParent ? state.parent : node;
      try {
        const groups = new Set(options.keys({ id: node, payload }));
        if (groups.size === 0) continue;
        const key = defaultKey(clampRank(options.rank({ id: node, payload })), target);
        const data = options.data
          ? JSON.stringify(options.data({ id: node, payload }) ?? null)
          : null;
        for (const group of groups) {
          // Tagged by what it stands for, so switching `reference` never reuses one.
          const reference = viewId(byParent ? 'parent-ref' : 'ref', group, target);
          next.set(`${node}\u0000${group}`, { node, group, defaultKey: key, data, reference });
        }
      } catch (error) {
        // A payload this device cannot map joins no group until the node or the version changes.
        report(error, node);
      }
    }
    const affected = new Set<string>();
    const gone = [...old].filter(([id]) => !next.has(id)).map(([, row]) => row);
    const changed = [...next].filter(([id, row]) => {
      const was = old.get(id);
      return (
        !was ||
        was.defaultKey !== row.defaultKey ||
        was.data !== row.data ||
        was.reference !== row.reference
      );
    });
    for (const row of gone) affected.add(row.group);
    // Report a group when a member joins it or moves in it, not when only its data changes, which
    // may happen on every edit.
    for (const [id, row] of changed) {
      const was = old.get(id);
      if (was?.defaultKey !== row.defaultKey || was.reference !== row.reference) {
        affected.add(row.group);
      }
    }
    // Write before deleting, so a concurrent read never misses a member that stays.
    if (changed.length > 0) {
      await recordGroups(changed.map(([, row]) => row.group));
      const rows = changed.map(([, { node, group, defaultKey, data, reference }]) => [
        node,
        group,
        defaultKey,
        data,
        reference,
      ]);
      await runner.run(
        `INSERT OR REPLACE INTO ${MEMBERS} (view, node, group_key, default_key, data, reference)
         SELECT ?1, unhex(value->>0), value->>1, unhex(value->>2), value->>3, unhex(value->>4)
         FROM json_each(?2)`,
        [name, JSON.stringify(rows)],
      );
    }
    if (gone.length > 0) {
      await runner.run(
        `DELETE FROM ${MEMBERS} WHERE view = ?1
         AND (node, group_key) IN (SELECT unhex(value->>0), value->>1 FROM json_each(?2))`,
        [name, JSON.stringify(gone.map((row) => [row.node, row.group]))],
      );
    }
    return affected;
  };

  /** The recorded groups whose node is one of `nodes` or above one, deleted ones included. */
  const groupsAbove = async (nodes: Set<string>): Promise<string[]> =>
    JSON.parse(
      (await runner.getText(
        `WITH RECURSIVE up(node, parent) AS (
           SELECT node, parent FROM tree_nodes WHERE node IN (SELECT unhex(value) FROM json_each(?1))
           UNION
           SELECT t.node, t.parent FROM up JOIN tree_nodes t ON t.node = up.parent
         )
         SELECT json_group_array(DISTINCT g.group_key)
         FROM up JOIN ${GROUPS} g ON g.group_node = up.node AND g.view = ?2`,
        [JSON.stringify([...nodes]), name],
      )) ?? '[]',
    ) as string[];

  /** Applies one event and returns the groups whose members or order it changed. */
  const apply = async (event: MaterializationEvent): Promise<string[]> => {
    const nodes = new Set<string>();
    const viewNodes = new Set<string>();
    for (const change of event.changes) {
      if (change.node.startsWith(VIEW_NODE_PREFIX)) viewNodes.add(change.node);
      else nodes.add(change.node);
      // A move also changes the group the node left. When the node went under a node whose insert
      // has not arrived, the walk up from it ends there, so that group is the only one found.
      const from = 'parentBefore' in change ? change.parentBefore : null;
      if (from?.startsWith(VIEW_NODE_PREFIX)) viewNodes.add(from);
    }
    const affected = nodes.size > 0 ? await index(nodes) : new Set<string>();
    if (viewNodes.size > 0) for (const group of await groupsAbove(viewNodes)) affected.add(group);
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
        await runner.run(`DELETE FROM ${MEMBERS} WHERE view = ?1`, [name]);
      },
      onError: (error) => report(error),
    },
  );

  /** Reads each group's members and, if it is ordered, the live nodes of its subtree. */
  const layouts = async (groups: readonly string[]): Promise<Map<string, Layout>> => {
    const out = new Map<string, Layout>();
    const requested = [...new Set(groups)];
    if (requested.length === 0) return out;
    const rows = JSON.parse(
      (await runner.getText(
        `SELECT json_group_array(json_array(m.group_key, lower(hex(m.node)), lower(hex(m.default_key)),
                                            m.data, lower(hex(m.reference))))
         FROM ${MEMBERS} m JOIN tree_nodes t
           ON t.node = m.node AND t.tombstone = 0 AND ${inTreeSql('t')}
         WHERE m.view = ?1 AND m.group_key IN (SELECT value FROM json_each(?2))`,
        [name, JSON.stringify(requested)],
      )) ?? '[]',
    ) as [string, string, string, string | null, string][];
    // A group node counts while it is in the tree, deleted or not; below it, only live nodes do.
    const subtree = JSON.parse(
      (await runner.getText(
        `WITH RECURSIVE sub(node, parent, order_key) AS (
           SELECT node, parent, order_key FROM tree_nodes
           WHERE node IN (SELECT unhex(value) FROM json_each(?1)) AND parent IS NOT NULL
           UNION
           SELECT t.node, t.parent, t.order_key FROM tree_nodes t JOIN sub ON t.parent = sub.node
           WHERE t.tombstone = 0
         )
         SELECT json_group_array(json_array(lower(hex(sub.node)), lower(hex(sub.parent)),
                                            lower(hex(sub.order_key)), lower(hex(p.payload))))
         FROM sub LEFT JOIN tree_payload p ON p.node = sub.node`,
        [JSON.stringify(requested.map(groupNode))],
      )) ?? '[]',
    ) as [string, string, string, string][];

    const inTree = new Set<string>();
    const childrenOf = new Map<string, [string, string, string][]>();
    for (const [node, parent, key, payload] of subtree) {
      inTree.add(node);
      const siblings = childrenOf.get(parent) ?? [];
      siblings.push([node, key, payload]);
      childrenOf.set(parent, siblings);
    }
    const byGroup = new Map<string, Membership[]>();
    for (const [group, node, defaultKey, data, reference] of rows) {
      const list = byGroup.get(group) ?? [];
      list.push({ node, group, defaultKey, data, reference });
      byGroup.set(group, list);
    }

    for (const group of requested) {
      const byReference = new Map<string, Membership[]>();
      const sorted = (byGroup.get(group) ?? []).sort(
        (a, b) => compareHex(a.defaultKey, b.defaultKey) || compareHex(a.node, b.node),
      );
      for (const member of sorted) {
        const shared = byReference.get(member.reference) ?? [];
        shared.push(member);
        byReference.set(member.reference, shared);
      }
      const layout: Layout = {
        group,
        groupNode: groupNode(group),
        ordered: inTree.has(groupNode(group)),
        top: [],
        nodes: new Map(),
        references: new Map(),
      };
      const build = (node: string, parent: Slot | null): Slot[] =>
        (childrenOf.get(node) ?? [])
          .map(([child, key, payload]) => {
            const folder = isFolder(child);
            const slot: Slot = {
              node: child,
              key,
              folder,
              payload: folder && payload ? hexToBytes(payload) : null,
              members: folder ? [] : (byReference.get(child) ?? []),
              pending: false,
              parent,
              children: [],
            };
            layout.nodes.set(child, slot);
            slot.children = build(child, slot);
            return slot;
          })
          .sort(slotOrder);
      if (layout.ordered) layout.top = build(layout.groupNode, null);
      for (const [reference, shared] of byReference) {
        if (layout.nodes.has(reference)) continue;
        layout.top.push({
          node: reference,
          key: shared[0]!.defaultKey,
          folder: false,
          payload: null,
          members: shared,
          pending: true,
          parent: null,
          children: [],
        });
      }
      layout.top.sort(slotOrder);
      for (const slot of [...layout.nodes.values(), ...layout.top]) {
        for (const member of slot.members) layout.references.set(member.node, slot);
      }
      out.set(group, layout);
    }
    return out;
  };

  const items = (slots: Slot[]): TreecrdtViewItem[] =>
    slots.flatMap((slot): TreecrdtViewItem[] => {
      if (slot.folder) {
        return [
          { kind: 'folder', id: slot.node, payload: slot.payload, children: items(slot.children) },
        ];
      }
      // A reference whose member left: its children show in its place.
      if (slot.members.length === 0) return items(slot.children);
      return slot.members.map((member, i) => ({
        kind: 'member' as const,
        id: member.node,
        ...parseData(member.data),
        children: i === 0 ? items(slot.children) : [],
      }));
    });

  const flatten = (slots: Slot[], out: TreecrdtViewMember[]) => {
    for (const slot of slots) {
      for (const member of slot.members) {
        out.push({ id: member.node, placed: !slot.pending, ...parseData(member.data) });
      }
      flatten(slot.children, out);
    }
    return out;
  };

  const members = async (groups: readonly string[]) => {
    const out = new Map<string, TreecrdtViewMember[]>();
    for (const [group, layout] of await layouts(groups)) {
      const list = flatten(layout.top, []);
      if (list.length > 0) out.set(group, list);
    }
    return out;
  };

  const tree = async (group: string) => items((await layouts([group])).get(group)!.top);

  /** The slot of a member or folder of the group. */
  const find = (layout: Layout, item: string): Slot | undefined => {
    const reference = layout.references.get(item);
    if (reference) return reference;
    const slot = layout.nodes.get(item);
    return slot?.folder ? slot : undefined;
  };

  /**
   * Where `slot` (null: a new node) goes for `position`: under `container` (null: the top level,
   * which the group node holds along with the references the walk does not reach), right after
   * `left` and before `right`, null for an end. Null when the slot already shows there.
   */
  const dropPoint = (layout: Layout, slot: Slot | null, position: TreecrdtViewPosition) => {
    // A parent no longer in the group means the top level.
    const parent = position.parent == null ? null : (find(layout, position.parent) ?? null);
    let before = position.before == null ? null : (find(layout, position.before) ?? null);
    // Before itself, or a member that shares its reference: where it is.
    if (slot && before === slot) return null;
    // A `before` no longer in `parent` means last. One under a reference whose member left shows
    // in that reference's place, so the item goes under that reference too.
    if (before && shownParent(before) !== parent) before = null;
    const container = before ? before.parent : parent;
    for (let up = container; up; up = up.parent) {
      if (up === slot) throw new Error('TreeCRDT view: an item cannot go inside itself');
    }
    // Already there: under `parent`, right before `before` in the order shown.
    if (slot && shownParent(slot) === parent && nextShown(layout.top, slot) === before) return null;
    const siblings = container ? container.children : layout.top;
    const rest = siblings.filter((sibling) => sibling !== slot);
    const left = rest[(before ? rest.indexOf(before) : rest.length) - 1] ?? null;
    return { container, left, right: before };
  };

  const writer = client.local.forReplica(options.replica);
  const writeOptions = () => options.writeOptions?.();

  /** Those of `nodes` that are in the tree, deleted ones included. */
  const inTreeNodes = async (nodes: string[]): Promise<Set<string>> =>
    new Set(
      JSON.parse(
        (await runner.getText(
          `SELECT json_group_array(lower(hex(node))) FROM tree_nodes
           WHERE node IN (SELECT unhex(value) FROM json_each(?1)) AND parent IS NOT NULL`,
          [JSON.stringify(nodes)],
        )) ?? '[]',
      ) as string[],
    );

  /**
   * Those of `nodes` that are in the tree, each with whether a deletion hides it under `groupNode`:
   * it, or a node between it and the group node, is deleted. The walk misses any other because a
   * node above it has not arrived, such as a folder a peer moved it into, or the group node.
   */
  const locate = async (nodes: string[], groupNode: string): Promise<Map<string, boolean>> => {
    if (nodes.length === 0) return new Map();
    const rows = JSON.parse(
      (await runner.getText(
        `WITH RECURSIVE up(start, parent, deleted) AS (
           SELECT node, parent, tombstone <> 0 FROM tree_nodes
           WHERE node IN (SELECT unhex(value) FROM json_each(?1)) AND parent IS NOT NULL
           UNION
           SELECT up.start, t.parent, up.deleted OR t.tombstone <> 0
           FROM up JOIN tree_nodes t ON t.node = up.parent
           WHERE up.parent <> unhex(?2)
         )
         SELECT json_group_array(json_array(lower(hex(start)), hidden)) FROM (
           SELECT start, MAX(parent = unhex(?2) AND deleted) AS hidden FROM up GROUP BY start
         )`,
        [JSON.stringify(nodes), groupNode],
      )) ?? '[]',
    ) as [string, number | null][];
    return new Map(rows.map(([node, hidden]) => [node, hidden === 1]));
  };

  /**
   * Writes what an item needs before it goes under `container` between `neighbours`: if the group
   * is unordered, the view root, the group node and every member's reference where it lists; else
   * the container's and the neighbours' references if they are not in the tree, so they stay where
   * the item was placed against them. These are create-only inserts at keys all devices derive
   * alike, so concurrent ones agree and none moves a node someone placed. A container that a
   * deletion hides is moved back to hold the item. `except`, the item being placed, is left to its
   * own write.
   */
  const prepare = async (
    layout: Layout,
    container: Slot | null,
    neighbours: (Slot | null)[],
    except: Slot | null,
    ops: Operation[],
  ) => {
    await recordGroups([layout.group]);
    const create = async (parent: string, node: string, placement: TreecrdtSqlitePlacement) => {
      const op = await writer.insertIfAbsent(parent, node, placement, null, writeOptions());
      if (op) ops.push(op);
      return op;
    };
    /** Creates the slot's reference at its default key, and lists it at the key it gets. */
    const createReference = async (slot: Slot) => {
      const op = await create(layout.groupNode, slot.node, at(slot.key));
      // 'at' appends the reference's id to the key, and an item placed after it goes by that key.
      if (op?.kind.type === 'insert') slot.key = bytesToHex(op.kind.orderKey);
      return op !== null;
    };
    const needed = [container, ...neighbours].filter((slot): slot is Slot => !!slot?.pending);
    const located = await locate(
      needed.map((slot) => slot.node),
      layout.groupNode,
    );
    if (!layout.ordered) {
      await create(ROOT_NODE_ID_HEX, root, anywhere);
      await create(root, layout.groupNode, anywhere);
      for (const slot of layout.top) {
        if (slot !== except && (await createReference(slot))) slot.pending = false;
      }
    }
    for (const slot of needed) {
      if (!slot.pending) continue;
      const hidden = located.get(slot.node);
      if (hidden === undefined) {
        // Writes nothing if a peer created it meanwhile, which leaves it where the peer put it.
        await createReference(slot);
      } else if (hidden && slot === container) {
        // Only a move shows a deleted container again, as it must to hold the item.
        ops.push(await writer.move(slot.node, layout.groupNode, at(slot.key), writeOptions()));
      }
      // Any other waits on a node that has not arrived. Moving it would undo the peer's move that
      // put it there, so the item goes under it as it is, and shows with it.
    }
  };

  const placeNow = async (group: string, item: string, position: TreecrdtViewPosition) => {
    // Place against the order the index shows once it has applied everything committed.
    await subscription.settled();
    const layout = (await layouts([group])).get(group)!;
    const slot = find(layout, item);
    // The member left the group meanwhile, such as by a concurrent edit, or the folder is gone.
    if (!slot) return [];
    const drop = dropPoint(layout, slot, position);
    if (!drop) return [];
    const inTree = slot.pending && (await inTreeNodes([slot.node])).has(slot.node);
    const ops: Operation[] = [];
    await prepare(layout, drop.container, [drop.left, drop.right], slot, ops);
    const parent = drop.container?.node ?? layout.groupNode;
    const placement = placementBetween(drop.container, drop.left, drop.right);
    ops.push(
      slot.pending && !inTree
        ? // The drag creates the reference it finds missing with a plain insert, which wins over
          // every automatic create of it, wherever that falls in the operation order.
          await writer.insert(parent, slot.node, placement, null, writeOptions())
        : await writer.move(slot.node, parent, placement, writeOptions()),
    );
    return ops;
  };

  const repairNow = async (groups?: readonly string[]) => {
    await subscription.settled();
    if (groups?.length === 0) return [];
    // One row per missing reference of an ordered group, at the least default key of its members.
    const missing = `
      JOIN tree_nodes t ON t.node = m.node AND t.tombstone = 0 AND ${inTreeSql('t')}
      LEFT JOIN tree_nodes r ON r.node = m.reference`;
    const pick = `m.group_key AS group_key, lower(hex(m.reference)) AS reference,
                  lower(hex(MIN(m.default_key))) AS key`;
    const rows = JSON.parse(
      (await (groups
        ? runner.getText(
            `SELECT json_group_array(json_array(group_key, reference, key)) FROM (
               SELECT ${pick} FROM json_each(?2) g
               JOIN tree_nodes gn ON gn.node = unhex(g.value->>1) AND gn.parent IS NOT NULL
               JOIN ${MEMBERS} m ON m.view = ?1 AND m.group_key = g.value->>0
               ${missing}
               WHERE r.parent IS NULL
               GROUP BY m.group_key, m.reference ORDER BY m.group_key, key
             )`,
            [name, JSON.stringify([...new Set(groups)].map((group) => [group, groupNode(group)]))],
          )
        : runner.getText(
            `SELECT json_group_array(json_array(group_key, reference, key)) FROM (
               SELECT ${pick} FROM tree_nodes gn
               JOIN ${GROUPS} g ON g.group_node = gn.node AND g.view = ?1
               JOIN ${MEMBERS} m ON m.view = ?1 AND m.group_key = g.group_key
               ${missing}
               WHERE gn.parent = ?2 AND r.parent IS NULL
               GROUP BY m.group_key, m.reference ORDER BY m.group_key, key
             )`,
            [name, hexToBytes(root)],
          ))) ?? '[]',
    ) as [string, string, string][];
    const ops: Operation[] = [];
    for (const [group, reference, key] of rows) {
      const op = await writer.insertIfAbsent(
        groupNode(group),
        reference,
        at(key),
        null,
        writeOptions(),
      );
      if (op) ops.push(op);
    }
    return ops;
  };

  const createFolderNow = async (
    group: string,
    { payload = null, ...position }: TreecrdtViewPosition & { payload?: Uint8Array | null },
  ) => {
    await subscription.settled();
    const layout = (await layouts([group])).get(group)!;
    const drop = dropPoint(layout, null, position)!;
    const ops: Operation[] = [];
    await prepare(layout, drop.container, [drop.left, drop.right], null, ops);
    const id = folderId();
    const parent = drop.container?.node ?? layout.groupNode;
    const placement = placementBetween(drop.container, drop.left, drop.right);
    ops.push(await writer.insert(parent, id, placement, payload, writeOptions()));
    return { id, ops };
  };

  const removeFolderNow = async (group: string, folder: string) => {
    await subscription.settled();
    const layout = (await layouts([group])).get(group)!;
    const slot = layout.nodes.get(folder);
    if (!slot?.folder) return [];
    const siblings = slot.parent ? slot.parent.children : layout.top;
    const index = siblings.indexOf(slot);
    const [before, after] = [siblings[index - 1] ?? null, siblings[index + 1] ?? null];
    const ops: Operation[] = [];
    await prepare(layout, slot.parent, [before, after], null, ops);
    const parent = slot.parent?.node ?? layout.groupNode;
    // Each item goes right after the one before, between the folder's neighbours.
    let left: { key: string } | null = before;
    for (const child of slot.children) {
      const placement = placementBetween(slot.parent, left, after);
      const op = await writer.move(child.node, parent, placement, writeOptions());
      ops.push(op);
      if (op.kind.type === 'move') left = { key: bytesToHex(op.kind.orderKey) };
    }
    ops.push(await writer.delete(folder, writeOptions()));
    return ops;
  };

  // One write at a time: a second drop is placed against the order the first left, which is the
  // order the user saw when making it.
  let writing: Promise<unknown> = Promise.resolve();
  const serially = <T>(work: () => Promise<T>): Promise<T> => {
    const next = writing.then(work);
    writing = next.catch(() => undefined);
    return next;
  };

  return {
    members,
    tree,
    place: (group, item, position = {}) => serially(() => placeNow(group, item, position)),
    repair: (groups) => serially(() => repairNow(groups)),
    createFolder: (group, opts = {}) => serially(() => createFolderNow(group, opts)),
    removeFolder: (group, folder) => serially(() => removeFolderNow(group, folder)),
    onChange: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    subscription,
  };
}
