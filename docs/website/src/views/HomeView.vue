<script setup lang="ts">
import { ref, onMounted, nextTick } from "vue";
import { loadDoc } from "../composables/useDocs";
import { renderMarkdown } from "../composables/useMarkdown";

interface FeatureCard {
  icon: string;
  title: string;
  desc: string;
  to: string;
}

const cards: FeatureCard[] = [
  {
    icon: "🏗️",
    title: "Architecture",
    desc: "Module boundaries across search, core, CLI, and ingestion.",
    to: "/ARCHITECTURE",
  },
  {
    icon: "🗺️",
    title: "Roadmap",
    desc: "What's shipped and what's in progress across every subsystem.",
    to: "/moon/ROADMAP",
  },
  {
    icon: "🔍",
    title: "Search",
    desc: "The recommendation and semantic-search subsystem roadmap.",
    to: "/moon/roadmaps/search",
  },
  {
    icon: "⚙️",
    title: "Core",
    desc: "The core recommendation-engine roadmap.",
    to: "/moon/roadmaps/core",
  },
  {
    icon: "📥",
    title: "Ingestion",
    desc: "The data-ingestion pipeline roadmap.",
    to: "/moon/roadmaps/ingestion",
  },
  {
    icon: "🔬",
    title: "Research",
    desc: "Building a Smart Recommendation Engine — the literature review behind the design.",
    to: "/research/Building a Smart Recommendation Engine",
  },
];

const state = ref<"loading" | "ok" | "notfound">("loading");
const html = ref("");

async function renderMermaid() {
  const nodes = document.querySelectorAll<HTMLElement>(".home-readme .mermaid");
  if (!nodes.length) return;
  const mermaid = (await import("mermaid")).default;
  mermaid.initialize({
    startOnLoad: false,
    theme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "default",
    securityLevel: "loose",
  });
  try {
    await mermaid.run({ nodes: Array.from(nodes) });
  } catch (e) {
    console.warn("mermaid render failed", e);
  }
}

async function load() {
  const raw = await loadDoc("README.md");
  if (raw === null) {
    state.value = "notfound";
    return;
  }
  html.value = renderMarkdown(raw);
  state.value = "ok";
  await nextTick();
  renderMermaid();
}

onMounted(load);
</script>

<template>
  <div class="home">
    <section class="hero-section">
      <div class="hero-overlay" />
      <div class="hero-content">
        <span class="badge">Documentation</span>
        <h1>Content-Recommendation-Engine</h1>
        <p class="hero-desc">
          A search and recommendation subsystem for the Image-Toolkit ecosystem — semantic
          search, ranking, and ingestion pipelines for surfacing related media.
        </p>
        <div class="hero-actions">
          <router-link to="/moon/ROADMAP" class="btn btn-primary">View the Roadmap</router-link>
          <router-link to="/ARCHITECTURE" class="btn btn-secondary">Read the Architecture</router-link>
          <a
            class="btn btn-secondary"
            href="https://github.com/ACFHarbinger/Image-Toolkit"
            target="_blank"
            rel="noopener noreferrer"
          >
            Parent Project ↗
          </a>
        </div>
      </div>
    </section>

    <section class="feature-grid-section">
      <div class="feature-grid">
        <router-link v-for="c in cards" :key="c.title" :to="c.to" class="feature-card panel-glass">
          <span class="feature-icon">{{ c.icon }}</span>
          <h3>{{ c.title }}</h3>
          <p>{{ c.desc }}</p>
        </router-link>
      </div>
    </section>

    <section class="home-readme">
      <div v-if="state === 'loading'" class="doc-loading">Loading…</div>
      <div v-else-if="state === 'ok'" class="markdown-body" v-html="html" />
    </section>
  </div>
</template>

<style scoped>
.hero-section {
  position: relative;
  padding: 4.5rem 2rem 3rem;
  overflow: hidden;
}
.hero-overlay {
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 20% 15%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 60%),
    radial-gradient(circle at 85% 30%, color-mix(in srgb, var(--accent-2) 12%, transparent), transparent 55%);
  pointer-events: none;
}
.hero-content {
  position: relative;
  z-index: 1;
  max-width: 720px;
  margin: 0 auto;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.hero-content h1 {
  font-family: var(--font-display);
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin: 1.25rem 0 1rem;
}
.hero-desc {
  font-size: 1.05rem;
  color: var(--text-muted);
  line-height: 1.65;
  margin: 0 0 2rem;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.85rem;
}

.feature-grid-section {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 2rem 3rem;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem;
}
.feature-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 1.5rem;
  color: var(--text);
}
.feature-icon {
  font-size: 1.6rem;
  margin-bottom: 0.75rem;
}
.feature-card h3 {
  margin: 0 0 0.5rem;
  font-size: 1.05rem;
  font-weight: 700;
}
.feature-card p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--text-muted);
  line-height: 1.5;
}

.home-readme {
  max-width: 900px;
  margin: 0 auto;
  padding: 1rem 2rem 6rem;
  border-top: 1px solid var(--border);
  padding-top: 3rem;
}
.doc-loading {
  text-align: center;
  color: var(--text-muted);
  padding: 3rem 0;
}

@media (max-width: 640px) {
  .hero-content h1 {
    font-size: 2.25rem;
  }
}
</style>
