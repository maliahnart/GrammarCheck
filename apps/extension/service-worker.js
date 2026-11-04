chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({id: "grammarcheck-selection", title: "Kiểm tra tiếng Việt với GrammarCheck", contexts: ["selection"]});
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "grammarcheck-selection" || !info.selectionText) return;
  await chrome.storage.session.set({pendingText: info.selectionText});
  await chrome.tabs.create({url: chrome.runtime.getURL("index.html")});
});
