import type { Operation, ReplicaId } from './index.js';
import type { TreecrdtSqlitePlacement } from './sqlite.js';
import type {
  BoundEngineLocal,
  EngineHistory,
  EngineLocal,
  EngineTree,
  LocalDeleteAction,
  LocalEditAction,
  LocalEditPlan,
  LocalMoveAction,
  LocalPayloadAction,
  LocalWriteOptions,
  OperationEdit,
} from './engine.js';

export type DeleteAction = LocalDeleteAction;

export type MoveAction = LocalMoveAction;

export type PayloadAction = LocalPayloadAction;

/**
 * A forward local write that reverses a previously captured local write.
 *
 * These are intentionally not oplog rewinds. Applying a plan mints new local ops, so undo/redo
 * remains mergeable with remote peers.
 */
export type Action = LocalEditAction;

export type Plan = LocalEditPlan;

export type Target = {
  tree: EngineTree;
  local: EngineLocal;
};

export type HistoryTarget = Target & {
  history: EngineHistory;
};

export type CapturedPlan<T> = {
  result: T;
  operations: Operation[];
  undo: Plan;
};

export type Captured<T> = {
  result: T;
  operations: Operation[];
};

export type { OperationEdit };

export type Undoable = {
  operations?: Operation[];
  undo: Plan;
};

export type Redoable = {
  operations?: Operation[];
  redo: Plan;
};

export type UndoResult = {
  operations: Operation[];
  redo: Plan;
};

export type RedoResult = {
  operations: Operation[];
  undo: Plan;
};

export type EditMode = 'force' | 'safe';

export type EditOptions = LocalWriteOptions & {
  /**
   * `force` always applies inverse operations. `safe` first checks that the current visible state
   * still matches the state produced by the captured edit, and rejects without writing on conflict.
   */
  mode?: EditMode;
};

export type EditConflictReason = 'exists' | 'parent' | 'orderKey' | 'payload';

export type EditConflict = {
  node: string;
  reason: EditConflictReason;
};

export class EditConflictError extends Error {
  readonly conflicts: EditConflict[];

  constructor(action: 'undo' | 'redo', conflicts: EditConflict[]) {
    super(`treecrdt: ${action} safe mode conflict`);
    this.name = 'EditConflictError';
    this.conflicts = conflicts;
  }
}

function cloneBytes(bytes: Uint8Array | null): Uint8Array | null {
  return bytes === null ? null : new Uint8Array(bytes);
}

function bytesEqual(a: Uint8Array | null | undefined, b: Uint8Array | null | undefined): boolean {
  if (a === null || a === undefined || b === null || b === undefined) return a === b;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

type ExpectedPostState = {
  exists?: boolean;
  parent?: string;
  orderKey?: Uint8Array;
  payload?: Uint8Array | null;
};

function expectedPostStates(operations: Operation[]): Map<string, ExpectedPostState> {
  const states = new Map<string, ExpectedPostState>();
  const stateFor = (node: string) => {
    let state = states.get(node);
    if (!state) {
      state = {};
      states.set(node, state);
    }
    return state;
  };

  for (const op of operations) {
    const kind = op.kind;
    switch (kind.type) {
      case 'insert': {
        const state = stateFor(kind.node);
        state.exists = true;
        state.parent = kind.parent;
        state.orderKey = kind.orderKey;
        state.payload = kind.payload === undefined ? null : cloneBytes(kind.payload);
        break;
      }
      case 'move': {
        const state = stateFor(kind.node);
        state.exists = true;
        state.parent = kind.newParent;
        state.orderKey = kind.orderKey;
        break;
      }
      case 'delete':
      case 'tombstone': {
        const state = stateFor(kind.node);
        state.exists = false;
        delete state.parent;
        delete state.orderKey;
        delete state.payload;
        break;
      }
      case 'payload': {
        const state = stateFor(kind.node);
        state.exists = true;
        state.payload = cloneBytes(kind.payload);
        break;
      }
    }
  }

  return states;
}

function splitEditOptions(opts?: EditOptions): {
  mode: EditMode;
  writeOptions?: LocalWriteOptions;
} {
  if (!opts) return { mode: 'force' };
  const { mode = 'force', ...writeOptions } = opts;
  return { mode, writeOptions };
}

function editOperations(edit: { operations?: Operation[] }): Operation[] {
  if (edit.operations) return edit.operations;
  throw new Error('treecrdt: safe edit mode requires captured operations');
}

async function assertSafePostState(
  target: Target,
  operations: Operation[],
  action: 'undo' | 'redo',
): Promise<void> {
  const expected = expectedPostStates(operations);
  const rowsByNode = new Map((await target.tree.dump()).map((row) => [row.node, row]));
  const conflicts: EditConflict[] = [];

  for (const [node, state] of expected) {
    const row = rowsByNode.get(node);
    const actualExists = row != null && !row.tombstone;

    if (state.exists !== undefined && actualExists !== state.exists) {
      conflicts.push({ node, reason: 'exists' });
      continue;
    }
    if (!actualExists) continue;

    if (state.parent !== undefined && row?.parent !== state.parent) {
      conflicts.push({ node, reason: 'parent' });
    }
    if (state.orderKey !== undefined && !bytesEqual(row?.orderKey, state.orderKey)) {
      conflicts.push({ node, reason: 'orderKey' });
    }
    if (state.payload !== undefined) {
      const actualPayload = await target.tree.getPayload(node);
      if (!bytesEqual(actualPayload, state.payload)) {
        conflicts.push({ node, reason: 'payload' });
      }
    }
  }

  if (conflicts.length > 0) {
    throw new EditConflictError(action, conflicts);
  }
}

async function visiblePlacementBefore(
  tree: EngineTree,
  parent: string,
  node: string,
): Promise<TreecrdtSqlitePlacement> {
  const siblings = await tree.children(parent);
  const index = siblings.indexOf(node);
  if (index < 0) {
    throw new Error(`treecrdt: cannot capture undo position for non-visible node ${node}`);
  }
  if (index === 0) return { type: 'first' };
  return { type: 'after', after: siblings[index - 1]! };
}

async function currentMoveUndoAction(tree: EngineTree, node: string): Promise<Action> {
  if (!(await tree.exists(node))) return { type: 'delete', node };

  const parent = await tree.parent(node);
  if (parent === null) {
    throw new Error(`treecrdt: cannot capture undo parent for root or missing node ${node}`);
  }
  return {
    type: 'move',
    node,
    parent,
    placement: await visiblePlacementBefore(tree, parent, node),
  };
}

/**
 * Capture the inverse plan eagerly while local writes are performed.
 *
 * This is useful when a caller wants undo metadata immediately, but it may read current
 * parent/sibling/payload state on the write path. App integrations that have `engine.history`
 * should prefer `capture(...)` and defer inversion until undo time.
 */
export async function capturePlan<T>(
  target: Target,
  replica: ReplicaId,
  fn: (local: BoundEngineLocal) => Promise<T>,
  defaults: LocalWriteOptions = {},
): Promise<CapturedPlan<T>> {
  const base = target.local.forReplica(replica, defaults);
  const operations: Operation[] = [];
  const actions: Action[] = [];

  const record = (op: Operation, action: Action) => {
    operations.push(op);
    actions.unshift(action);
  };

  const local: BoundEngineLocal = {
    insert: async (parent, node, placement, payload, opts) => {
      const op = await base.insert(parent, node, placement, payload, opts);
      record(op, { type: 'delete', node });
      return op;
    },
    move: async (node, newParent, placement, opts) => {
      const undo = await currentMoveUndoAction(target.tree, node);
      const op = await base.move(node, newParent, placement, opts);
      record(op, undo);
      return op;
    },
    delete: async (node, opts) => {
      const undo = await currentMoveUndoAction(target.tree, node);
      const op = await base.delete(node, opts);
      record(op, undo);
      return op;
    },
    payload: async (node, payload, opts) => {
      const previous = await target.tree.getPayload(node);
      const op = await base.payload(node, payload, opts);
      record(op, { type: 'payload', node, payload: cloneBytes(previous) });
      return op;
    },
  };

  const result = await fn(local);
  return { result, operations, undo: { actions } };
}

/**
 * Capture the local operations produced by the callback.
 *
 * This avoids inverse reads on the write path. Use `undo(...)` with a history-capable engine to
 * derive the inverse later from the operation log.
 */
export async function capture<T>(
  target: Target,
  replica: ReplicaId,
  fn: (local: BoundEngineLocal) => Promise<T>,
  defaults: LocalWriteOptions = {},
): Promise<Captured<T>> {
  const base = target.local.forReplica(replica, defaults);
  const operations: Operation[] = [];

  const local: BoundEngineLocal = {
    insert: async (parent, node, placement, payload, opts) => {
      const op = await base.insert(parent, node, placement, payload, opts);
      operations.push(op);
      return op;
    },
    move: async (node, newParent, placement, opts) => {
      const op = await base.move(node, newParent, placement, opts);
      operations.push(op);
      return op;
    },
    delete: async (node, opts) => {
      const op = await base.delete(node, opts);
      operations.push(op);
      return op;
    },
    payload: async (node, payload, opts) => {
      const op = await base.payload(node, payload, opts);
      operations.push(op);
      return op;
    },
  };

  const result = await fn(local);
  return { result, operations };
}

export async function applyToLocal(
  local: BoundEngineLocal,
  plan: Plan,
  opts?: LocalWriteOptions,
): Promise<Operation[]> {
  const operations: Operation[] = [];
  for (const action of plan.actions) {
    switch (action.type) {
      case 'delete':
        operations.push(await local.delete(action.node, opts));
        break;
      case 'move':
        operations.push(await local.move(action.node, action.parent, action.placement, opts));
        break;
      case 'payload':
        operations.push(await local.payload(action.node, cloneBytes(action.payload), opts));
        break;
    }
  }
  return operations;
}

export function apply(
  target: Target,
  replica: ReplicaId,
  plan: Plan,
  opts?: LocalWriteOptions,
): Promise<Operation[]> {
  return applyToLocal(target.local.forReplica(replica), plan, opts);
}

export async function undo(
  target: HistoryTarget,
  replica: ReplicaId,
  edit: OperationEdit,
  opts?: EditOptions,
): Promise<UndoResult>;
export async function undo(
  target: Target,
  replica: ReplicaId,
  edit: Undoable,
  opts?: EditOptions,
): Promise<UndoResult>;
export async function undo(
  target: Target | HistoryTarget,
  replica: ReplicaId,
  edit: OperationEdit | Undoable,
  opts?: EditOptions,
): Promise<UndoResult> {
  const { mode, writeOptions } = splitEditOptions(opts);
  if (mode === 'safe') {
    await assertSafePostState(target, editOperations(edit), 'undo');
  }
  const undo =
    'undo' in edit ? edit.undo : await (target as Partial<HistoryTarget>).history?.invert(edit);
  if (!undo) {
    throw new Error('treecrdt: history inversion is not implemented by this engine');
  }
  const applied = await capturePlan(target, replica, (local) =>
    applyToLocal(local, undo, writeOptions),
  );
  return { operations: applied.operations, redo: applied.undo };
}

export async function redo(
  target: Target,
  replica: ReplicaId,
  edit: Redoable,
  opts?: EditOptions,
): Promise<RedoResult> {
  const { mode, writeOptions } = splitEditOptions(opts);
  if (mode === 'safe') {
    await assertSafePostState(target, editOperations(edit), 'redo');
  }
  const applied = await capturePlan(target, replica, (local) =>
    applyToLocal(local, edit.redo, writeOptions),
  );
  return { operations: applied.operations, undo: applied.undo };
}

export const edits = {
  apply,
  applyToLocal,
  capture,
  capturePlan,
  redo,
  undo,
};
