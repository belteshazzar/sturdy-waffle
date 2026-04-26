'use strict';

function decodeHtmlEntities(text) {
  return text
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, '\'');
}

function stripHtml(html) {
  if (!html || typeof html !== 'string') return '';
  let text = html;
  text = text.replace(/<script[\s\S]*?<\/script>/gi, ' ');
  text = text.replace(/<style[\s\S]*?<\/style>/gi, ' ');
  text = text.replace(/<h[1-6][^>]*>([\s\S]*?)<\/h[1-6]>/gi, (_, heading) => {
    const cleanHeading = stripHtml(heading).trim();
    return `\n== ${cleanHeading} ==\n`;
  });
  text = text.replace(/<br\s*\/?>/gi, '\n');
  text = text.replace(/<\/p>/gi, '\n');
  text = text.replace(/<[^>]+>/g, ' ');
  text = decodeHtmlEntities(text);
  return text;
}

function stripWikitext(text) {
  if (!text || typeof text !== 'string') return '';
  let cleaned = text;
  cleaned = cleaned.replace(/<ref[^>]*>[\s\S]*?<\/ref>/gi, ' ');
  cleaned = cleaned.replace(/<!--[\s\S]*?-->/g, ' ');
  cleaned = cleaned.replace(/\{\{[^{}]*\}\}/g, ' ');
  cleaned = cleaned.replace(/\[\[File:[^\]]+\]\]/gi, ' ');
  cleaned = cleaned.replace(/\[\[Image:[^\]]+\]\]/gi, ' ');
  cleaned = cleaned.replace(/\[\[([^|\]]+)\|([^\]]+)\]\]/g, '$2');
  cleaned = cleaned.replace(/\[\[([^\]]+)\]\]/g, '$1');
  cleaned = cleaned.replace(/'''+/g, '');
  cleaned = cleaned.replace(/''/g, '');
  cleaned = cleaned.replace(/<[^>]+>/g, ' ');
  return cleaned;
}

function segmentSections(text, { defaultTitle = 'Summary' } = {}) {
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  const sections = [];
  let current = { title: defaultTitle, text: '' };
  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const headingMatch = trimmed.match(/^==+\s*(.+?)\s*==+$/);
    if (headingMatch) {
      if (current.text.trim()) sections.push({ ...current, text: current.text.trim() });
      current = { title: headingMatch[1].trim(), text: '' };
      return;
    }
    current.text += (current.text ? ' ' : '') + trimmed;
  });
  if (current.text.trim()) sections.push({ ...current, text: current.text.trim() });
  return sections;
}

function splitSentences(text) {
  if (!text) return [];
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const sentences = cleaned.split(/(?<=[.!?])\s+(?=[A-Z0-9])/);
  return sentences.map(sentence => sentence.trim()).filter(Boolean);
}

function segmentWikipediaText({ extract, html, wikitext }) {
  let baseText = extract || '';
  if (!baseText && html) {
    baseText = stripHtml(html);
  }
  if (!baseText && wikitext) {
    baseText = stripWikitext(wikitext);
  }
  const sections = segmentSections(baseText);
  return sections.map(section => ({
    ...section,
    sentences: splitSentences(section.text),
  }));
}

module.exports = {
  stripHtml,
  stripWikitext,
  segmentSections,
  splitSentences,
  segmentWikipediaText,
};
