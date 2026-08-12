import { Database, GitBranch, Search } from "lucide-react";

export const site = {
  shortName: "CRE",
  name: "Content Recommendation Engine",
  eyebrow: "LOCAL-FIRST / DISCOVERY SYSTEM",
  badge: "VECTOR LAB / v1.0",
  heroTitle: ["Find the", "signal."],
  heroDescription: "A privacy-conscious recommendation engine combining vector representations, structured metadata, graph context, and explainable ranking for visual collections.",
  accent: "cyan",
  repository: "https://github.com/ACFHarbinger/Content-Recommendation-Engine",
  modules: [
    { number: "01", title: "Represent Content", text: "Turn visual assets, metadata, and entities into searchable representations.", detail: "Preserve provenance as content moves from ingestion into the retrieval graph.", action: "Read the architecture", href: "https://github.com/ACFHarbinger/Content-Recommendation-Engine/blob/main/docs/ARCHITECTURE.md", icon: Database },
    { number: "02", title: "Retrieve Context", text: "Combine semantic similarity with explicit filters and constraints.", detail: "Make recommendations useful without making the ranking process opaque.", action: "Inspect the system", href: "#pipeline", icon: Search },
    { number: "03", title: "Learn Carefully", text: "Use opt-in interaction signals to improve ranking while respecting local data.", detail: "Measure relevance and explain why a result appeared.", action: "Read the roadmap", href: "https://github.com/ACFHarbinger/Content-Recommendation-Engine/blob/main/docs/moon/ROADMAP.md", icon: GitBranch },
  ],
  stages: ["INGEST", "EMBED", "FILTER", "RANK", "EXPLAIN", "FEEDBACK"],
};
