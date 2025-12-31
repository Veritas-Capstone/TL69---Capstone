import { underlineSentences } from '@/helpers';

export default defineContentScript({
	matches: ['<all_urls>'],
	main() {
		browser.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
			// selects text from entire webpage
			if (message.type === 'GET_PAGE_TEXT') {
				await browser.storage.local.set({ selectedText: document.body.innerText });
			}

			// underlines sentences on webpage
			if (message.type === 'UNDERLINE_SELECTION') {
				const { sentences } = message;
				underlineSentences(sentences);
			}

			// highlight specific sentence
			if (message.type === 'HIGHLIGHT_TEXT') {
				const { idx, valid } = message;
				const spans = document.querySelectorAll(`span.underline-${idx}`);

				spans.forEach((span) => {
					(span as HTMLElement).style.backgroundColor = valid
						? 'rgba(74, 222, 128, 0.5)'
						: 'rgba(244, 63, 94, 0.5)';
				});

				const firstSpan = spans[0] as HTMLElement | undefined;
				if (firstSpan) {
					firstSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
				}
			}

			// clears highlights from webpage
			if (message.type === 'CLEAR_HIGHLIGHTS') {
				const underlinedSpans = document.querySelectorAll('span[class^="underline"]');

				underlinedSpans.forEach((span) => {
					(span as HTMLElement).style.backgroundColor = '';
				});
			}
			// clears underlines from webpage
			if (message.type === 'CLEAR_UNDERLINES') {
				const underlinedSpans = document.querySelectorAll('span[class^="underline-"]');

				underlinedSpans.forEach((span) => {
					const parent = span.parentNode;
					if (!parent) return;

					parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});
			}
			// clear selected text from webpage
			if (message.type === 'CLEAR_SELECTION') {
				window.getSelection()?.removeAllRanges();
			}
		});
	},
});
