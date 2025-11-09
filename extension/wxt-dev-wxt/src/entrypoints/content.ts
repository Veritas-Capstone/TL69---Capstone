export default defineContentScript({
	matches: ['<all_urls>'],
	main(ctx) {
		browser.runtime.onMessage.addListener(async (message) => {
			if (message.type === 'GET_PAGE_TEXT') {
				await browser.storage.local.set({ selectedText: document.body.innerText });
			}
		});
	},
});
