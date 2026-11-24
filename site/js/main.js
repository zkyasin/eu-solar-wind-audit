import { renderAnomalies } from "./anomalies.js";
import { renderContent } from "./content.js";
import { load } from "./data.js";
import { renderHero } from "./hero.js";
import { renderModel } from "./model.js";
import { renderModulation } from "./modulation.js";

const REPOSITORY = "https://github.com/zkyasin/eu-solar-wind-audit";

/** The masthead takes a background once the page scrolls under it. */
function watchScroll() {
  const masthead = document.getElementById("masthead");
  const update = () => masthead.toggleAttribute("data-scrolled", window.scrollY > 8);
  update();
  window.addEventListener("scroll", update, { passive: true });
}

/** The link of the chapter in view is marked in the masthead. */
function watchChapters() {
  const links = new Map(
    [...document.querySelectorAll(".masthead nav a")].map((link) => [link.hash.slice(1), link]),
  );
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const [id, link] of links) {
          if (id === entry.target.id) link.setAttribute("aria-current", "true");
          else link.removeAttribute("aria-current");
        }
      }
    },
    { rootMargin: "-45% 0px -50% 0px" },
  );
  for (const id of links.keys()) observer.observe(document.getElementById(id));
}

async function renderMeta() {
  const meta = await load("meta.json");
  const commit = document.createElement("a");
  commit.href = `${REPOSITORY}/commit/${meta.commit}`;
  commit.textContent = meta.commit.slice(0, 7);
  document
    .getElementById("meta")
    .replaceChildren(
      `Data exported on ${meta.exported} from commit `,
      commit,
      ", results of 2019–2025.",
    );
}

watchScroll();
watchChapters();
const chapters = [
  renderMeta,
  renderHero,
  renderModel,
  renderAnomalies,
  renderContent,
  renderModulation,
];
const results = await Promise.allSettled(chapters.map((render) => render()));
for (const result of results) {
  if (result.status === "rejected") console.error(result.reason);
}
