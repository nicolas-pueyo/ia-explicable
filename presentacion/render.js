// Renderiza slides.html a PDF y PNG (una imagen por diapositiva) con Chromium.
const { chromium } = require('/opt/node-tools/node_modules/playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
  await p.goto('file://' + path.resolve(__dirname, 'slides.html'));
  await p.waitForLoadState('networkidle');
  const slides = await p.$$('.slide');
  for (let i = 0; i < slides.length; i++) await slides[i].screenshot({ path: path.join(__dirname, `preview-${i + 1}.png`) });
  // detectar desbordes
  const over = await p.$$eval('.slide, .card', els => els.filter(e => e.scrollHeight > e.clientHeight + 2).map(e => (e.querySelector('h3,h1')||{}).textContent + ' ' + e.scrollHeight + '>' + e.clientHeight));
  console.log('overflow:', over);
  await p.pdf({ path: path.join(__dirname, 'presentacion.pdf'), width: '1280px', height: '720px', printBackground: true });
  await b.close();
})();
