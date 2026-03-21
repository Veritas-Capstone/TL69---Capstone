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
				const res = underlineSentences(sentences);
				sendResponse(res);
			}

			// highlight specific sentence
			if (message.type === 'HIGHLIGHT_TEXT') {
				const { idx, valid, category, label } = message;
				const spans = document.querySelectorAll(`span.underline-${idx}`);

				spans.forEach((span) => {
					if (category) {
						(span as HTMLElement).style.backgroundColor =
							category === 'Left-leaning'
								? 'rgba(59, 130, 246, 0.5)'
								: category === 'Right-leaning'
									? 'rgba(239, 68, 68, 0.5)'
									: 'rgba(139, 92, 246, 0.5)';
					} else if (label) {
						(span as HTMLElement).style.backgroundColor =
							label === 'SUPPORTED'
								? 'rgba(59, 130, 246, 0.5)'
								: label === 'REFUTED'
									? 'rgba(239, 68, 68, 0.5)'
									: 'rgba(100, 116, 139, 0.5)';
					} else {
						(span as HTMLElement).style.backgroundColor = valid
							? 'rgba(74, 222, 128, 0.5)'
							: 'rgba(244, 63, 94, 0.5)';
					}
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
