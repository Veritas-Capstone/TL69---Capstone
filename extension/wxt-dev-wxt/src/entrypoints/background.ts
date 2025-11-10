import { browser } from 'wxt/browser';

export default defineBackground(() => {
	browser.contextMenus.create({
		id: 'analyze-highlighted-text',
		title: 'Analyze Highlighted Text',
		contexts: ['selection'],
	});

	browser.contextMenus.onClicked.addListener(async (info, tab) => {
		if (info.menuItemId === 'analyze-highlighted-text') {
			// store the selected text, open popup
			await browser.storage.local.remove('result');
			await browser.storage.local.set({ selectedText: info.selectionText });
			await browser.action.openPopup();
		}
	});

	browser.runtime.onMessage.addListener(async (message) => {
		if (message.type === 'UNDERLINE_CLICKED') {
			// when clicking underlined text, open popup and highlight specific claim
			await browser.storage.local.set({ clickedText: normalizeSpaces(message.text) });
			await browser.action.openPopup();
		}
	});
});

function normalizeSpaces(str: string) {
	return str
		.replace(/\s+/g, ' ')
		.replace(/\u00A0/g, ' ')
		.trim();
}
