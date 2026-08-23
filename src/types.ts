// Shared shapes for the generated bundle. This file contains type declarations only, so the
// build drops it entirely.

export interface Heading {
  level: number;
  text: string;
  id: string;
}

export interface Chapter {
  slug: string;
  file: string;
  title: string;
  subtitle: string;
  minutes: number;
  index: boolean;
  html: string;
  headings: Heading[];
  text: string;
  words: number;
  links: number;
}

export interface BuildInfo {
  generated: string;
  chapters: number;
  words: number;
  links: number;
  markdownBytes: number;
  tool: string;
}

export interface Bundle {
  chapters: Chapter[];
  build: BuildInfo;
}

export interface SearchHit {
  chapter: Chapter;
  score: number;
  snippet: string;
}
