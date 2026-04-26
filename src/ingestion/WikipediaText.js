'use strict';

function decodeHtmlEntities(text) {
  if (!text) return '';
  const entityMap = {
    nbsp: ' ',
    amp: '&',
    lt: '<',
    gt: '>',
    quot: '"',
    '#39': '\'',
  };
  return text.replace(/&(nbsp|amp|lt|gt|quot|#39);/gi, (match, key) => {
    const normalized = key.toLowerCase();
    return entityMap[normalized] || match;
  });
}

function removeTagBlocks(text, tagName) {
  if (!text) return '';
  const lower = text.toLowerCase();
  const openTag = `<${tagName}`;
  const closeTag = `</${tagName}`;
  let idx = 0;
  let result = '';
  while (true) {
    const start = lower.indexOf(openTag, idx);
    if (start < 0) {
      result += text.slice(idx);
      break;
    }
    result += text.slice(idx, start);
    const closeStart = lower.indexOf(closeTag, start);
    if (closeStart < 0) break;
    const closeEnd = lower.indexOf('>', closeStart);
    if (closeEnd < 0) break;
    idx = closeEnd + 1;
  }
  return result;
}

function stripInlineTags(text) {
  if (!text) return '';
  let result = '';
  let inTag = false;
  for (const ch of text) {
    if (ch === '<') {
      inTag = true;
      continue;
    }
    if (ch === '>') {
      inTag = false;
      continue;
    }
    if (!inTag) result += ch;
  }
  return result;
}

function stripHtml(html) {
  if (!html || typeof html !== 'string') return '';
  let text = removeTagBlocks(html, 'script');
  text = removeTagBlocks(text, 'style');
  text = text.replace(/<h[1-6][^>]*>([\s\S]*?)<\/h[1-6]>/gi, (_, heading) => {
    const cleanHeading = stripHtml(heading).trim();
    return `\n== ${cleanHeading} ==\n`;
  });
  text = text.replace(/<br\s*\/?>/gi, '\n');
  text = text.replace(/<\/p>/gi, '\n');
  text = stripInlineTags(text);
  text = decodeHtmlEntities(text);
  return text;
}

function stripWikitext(text) {
  if (!text || typeof text !== 'string') return '';
  let cleaned = removeTagBlocks(text, 'ref');
  cleaned = removeHtmlComments(cleaned);
  cleaned = stripTemplates(cleaned);
  cleaned = stripWikiLinks(cleaned);
  cleaned = cleaned.split("'''").join('').split("''").join('');
  cleaned = stripInlineTags(cleaned);
  return cleaned;
}

function removeHtmlComments(text) {
  if (!text) return '';
  let result = '';
  let idx = 0;
  while (idx < text.length) {
    const start = text.indexOf('<!--', idx);
    if (start < 0) {
      result += text.slice(idx);
      break;
    }
    result += text.slice(idx, start);
    const end = text.indexOf('-->', start + 4);
    if (end < 0) break;
    idx = end + 3;
  }
  return result;
}

function stripTemplates(text) {
  if (!text) return '';
  let result = '';
  let depth = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '{' && next === '{') {
      depth += 1;
      i += 1;
      continue;
    }
    if (ch === '}' && next === '}' && depth > 0) {
      depth -= 1;
      i += 1;
      continue;
    }
    if (depth === 0) result += ch;
  }
  return result;
}

function stripWikiLinks(text) {
  if (!text) return '';
  let result = '';
  let idx = 0;
  while (idx < text.length) {
    if (text[idx] === '[' && text[idx + 1] === '[') {
      const end = text.indexOf(']]', idx + 2);
      if (end < 0) {
        result += text.slice(idx);
        break;
      }
      const content = text.slice(idx + 2, end);
      const lower = content.toLowerCase();
      if (!lower.startsWith('file:') && !lower.startsWith('image:')) {
        const pipeIdx = content.indexOf('|');
        const label = pipeIdx >= 0 ? content.slice(pipeIdx + 1) : content;
        result += label;
      }
      idx = end + 2;
      continue;
    }
    result += text[idx];
    idx += 1;
  }
  return result;
}

function segmentSections(text, { defaultTitle = 'Summary' } = {}) {
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  const sections = [];
  let current = { title: defaultTitle, text: '' };
  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const headingTitle = parseHeading(trimmed);
    if (headingTitle) {
      if (current.text.trim()) sections.push({ ...current, text: current.text.trim() });
      current = { title: headingTitle, text: '' };
      return;
    }
    current.text += (current.text ? ' ' : '') + trimmed;
  });
  if (current.text.trim()) sections.push({ ...current, text: current.text.trim() });
  return sections;
}

function parseHeading(line) {
  if (!line.startsWith('==') || !line.endsWith('==')) return null;
  let start = 0;
  while (line[start] === '=') start += 1;
  let end = line.length - 1;
  while (line[end] === '=') end -= 1;
  if (start < 2) return null;
  const title = line.slice(start, end + 1).trim();
  return title || null;
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
