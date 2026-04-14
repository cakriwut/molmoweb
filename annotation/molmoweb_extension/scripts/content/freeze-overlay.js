(() => {
  if (document.getElementById('__webolmoFreeze')) return;
  const overlay = document.createElement('div');
  overlay.id = '__webolmoFreeze';
  Object.assign(overlay.style, {
    position: 'fixed',
    inset: '0',
    zIndex: '2147483647',
    background: 'rgba(0,0,0,0)',
    cursor: 'wait',
    pointerEvents: 'all'
  });
  document.documentElement.appendChild(overlay);

  function handle(msg) {
    if (msg && msg.type === 'unfreeze') {
      overlay.remove();
      chrome.runtime.onMessage.removeListener(handle);
    }
  }

  chrome.runtime.onMessage.addListener(handle);
})(); 