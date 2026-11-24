---
'@treecrdt/interface': minor
---

Add `createTreecrdtView` (`@treecrdt/interface/views`) for the SQLite-backed engines. It groups nodes by keys the app derives from each node's payload, in an index each device keeps in a new `treecrdt_view_members` table instead of writing ops, and gives each group an order that replicates: `place(group, member, before)` writes one record node at a key between the neighbours shown, and members no one placed sort by `rank`. The first placement in a document adds the records' subtree under the root, which every replica then has, so code that walks the tree or listens to `onMaterialized` must skip nodes whose ids start with `VIEW_NODE_PREFIX`.
