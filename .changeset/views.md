---
'@treecrdt/interface': minor
---

Add `createTreecrdtView` (`@treecrdt/interface/views`) for the SQLite-backed engines: it groups nodes by keys the app derives from each node's payload, in an index each device keeps in its own tables instead of writing ops, and orders a group someone orders as a subtree of the document, with a reference node per member (or per member's parent) ordered as ordinary siblings, folders and nesting, written by `place`, `createFolder`, `removeFolder` and `repair`, whose automatic writes are create-only inserts at keys every device derives alike, so none moves a reference someone placed. **The first ordering adds a view root among the root's children on every replica, so everything that reads the root's children, walks the tree or listens to `onMaterialized` (lists, exports, sibling placement) must skip nodes whose ids start with `VIEW_NODE_PREFIX`, and a writer whose capability covers only a subtree cannot order a group.**
