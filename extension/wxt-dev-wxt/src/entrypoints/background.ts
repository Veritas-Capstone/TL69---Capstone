import { browser } from 'wxt/browser';

export default defineBackground(() => {
	browser.contextMenus.create({
		id: 'analyze-highlighted-text',
		title: 'Analyze Highlighted Text',
		contexts: ['selection'],
	});

	browser.contextMenus.onClicked.addListener(async (info) => {
		if (info.menuItemId === 'analyze-highlighted-text') {
			// store the selected text, open popup
			await browser.storage.local.set({ selectedText: info.selectionText });
			await browser.action.openPopup();
		}
	});
});
