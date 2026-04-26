'use strict';

const https = require('https');

function fetchJson(url, { userAgent }) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, { headers: { 'User-Agent': userAgent } }, res => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode && res.statusCode >= 400) {
          return reject(new Error(`Wikipedia request failed (${res.statusCode}) for ${url}`));
        }
        try {
          resolve(JSON.parse(body));
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

function fetchText(url, { userAgent }) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, { headers: { 'User-Agent': userAgent } }, res => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode && res.statusCode >= 400) {
          return reject(new Error(`Wikipedia request failed (${res.statusCode}) for ${url}`));
        }
        resolve(body);
      });
    });
    req.on('error', reject);
    req.end();
  });
}

async function fetchWikipediaArticle(title, opts = {}) {
  if (!title) throw new Error('fetchWikipediaArticle: title is required');
  const {
    language = 'en',
    includeHtml = false,
    includeWikitext = false,
    userAgent = 'sturdy-waffle-wikipedia-ingestor/1.0',
  } = opts;

  const encodedTitle = encodeURIComponent(title.replace(/ /g, '_'));
  const summaryUrl = `https://${language}.wikipedia.org/api/rest_v1/page/summary/${encodedTitle}`;
  const summary = await fetchJson(summaryUrl, { userAgent });

  let html = null;
  if (includeHtml) {
    const htmlUrl = `https://${language}.wikipedia.org/api/rest_v1/page/html/${encodedTitle}`;
    html = await fetchText(htmlUrl, { userAgent });
  }

  let wikitext = null;
  if (includeWikitext) {
    const wikitextUrl =
      `https://${language}.wikipedia.org/w/api.php?action=parse&format=json&formatversion=2` +
      `&prop=wikitext&page=${encodedTitle}`;
    const wikitextResponse = await fetchJson(wikitextUrl, { userAgent });
    wikitext = wikitextResponse?.parse?.wikitext || null;
  }

  return {
    title: summary.title || title,
    normalizedTitle: summary?.titles?.normalized || summary.title || title,
    displayTitle: summary?.titles?.display || summary.title || title,
    description: summary.description || null,
    extract: summary.extract || null,
    pageid: summary.pageid || null,
    contentUrls: summary.content_urls || null,
    language,
    html,
    wikitext,
  };
}

module.exports = {
  fetchWikipediaArticle,
};
