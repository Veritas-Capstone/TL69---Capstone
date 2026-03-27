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
			await browser.sidePanel.open({ tabId: tab.id });
			await browser.storage.local.clear();
			await browser.storage.local.set({ selectedText: info.selectionText });
			try {
				await browser.runtime.sendMessage({ type: 'CALL_MODEL' });
			} catch {
				// Panel will pick up selectedText from storage on mount
			}
		}
	});
	browser.runtime.onMessage.addListener(async (msg, sender) => {
		if (msg.type === 'OPEN_SIDEBAR') {
			if (browser.sidePanel) {
				await browser.sidePanel.open({
					tabId: sender.tab?.id ?? 0,
				});
			}
		}
	});
});
