import { browser } from 'wxt/browser';

export default defineBackground(() => {
	// creates analyze text button in right click menu
	browser.contextMenus.create({
		id: 'analyzeHighlightedText',
		title: 'Analyze Highlighted Text',
		contexts: ['selection'],
	});

	browser.contextMenus.onClicked.addListener(async (info, tab) => {
		if (info.menuItemId === 'analyzeHighlightedText') {
			if (!tab?.id) return;
			// open sidepanel, store selected text, call model on text
			await browser.sidePanel.open({ tabId: tab.id });
			await browser.storage.local.remove('storedResult');
			await browser.storage.local.set({ selectedText: info.selectionText });
			await browser.runtime.sendMessage({
				type: 'CALL_MODEL',
			});
		}
	});
});
