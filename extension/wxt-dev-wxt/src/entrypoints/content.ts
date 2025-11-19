export default defineContentScript({
	matches: ['<all_urls>'],
	main() {
		browser.runtime.onMessage.addListener(async (message) => {
			// selects text from entire webpage
			if (message.type === 'GET_PAGE_TEXT') {
				await browser.storage.local.set({ selectedText: document.body.innerText });
			}

			// underlines sentences on webpage
			if (message.type === 'UNDERLINE_SELECTION') {
				const { targets } = message;
				if (!targets || !Array.isArray(targets)) return;

				const walk = (node: Node) => {
					if (node.nodeType === Node.TEXT_NODE) {
						const parent = node.parentNode;
						if (!parent) return;

						let text = node.nodeValue || '';
						const frag = document.createDocumentFragment();
						let cursor = 0;

						while (cursor < text.length) {
							let matchIndex = text.length;
							let matchedTarget = '';

							for (const target of targets) {
								const normalizedText = normalizeSpaces(text.slice(cursor));
								const normalizedTarget = normalizeSpaces(target);

								const index = normalizedText.indexOf(normalizedTarget);
								if (index !== -1 && cursor + index < matchIndex) {
									matchIndex = cursor + index;
									matchedTarget = text.slice(cursor + index, cursor + index + normalizedTarget.length);
								}
							}

							if (!matchedTarget) {
								frag.append(document.createTextNode(text.slice(cursor)));
								break;
							}

							if (matchIndex > cursor) {
								frag.append(document.createTextNode(text.slice(cursor, matchIndex)));
							}

							const span = document.createElement('span');
							span.className = 'my-extension-underline';
							span.style.textDecoration = 'underline';
							span.style.textDecorationColor = message.valid ? 'rgb(74, 222, 128)' : 'red';
							span.style.textDecorationThickness = '2px';
							span.style.paddingTop = '4px';
							span.style.paddingBottom = '4px';
							span.style.lineHeight = '2px';
							span.textContent = matchedTarget;
							span.style.cursor = 'pointer';

							span.addEventListener('mouseenter', () => {
								browser.runtime.sendMessage({
									type: 'UNDERLINE_HOVER',
									text: normalizeSpaces(matchedTarget),
								});
							});

							span.addEventListener('mouseleave', () => {
								browser.runtime.sendMessage({
									type: 'UNDERLINE_HOVER',
									text: '',
								});
							});

							frag.append(span);
							cursor = matchIndex + matchedTarget.length;
						}

						parent.replaceChild(frag, node);
					} else if (node.nodeType === Node.ELEMENT_NODE) {
						const tag = (node as HTMLElement).tagName;
						if (tag === 'SCRIPT' || tag === 'STYLE') return;
						Array.from(node.childNodes).forEach(walk);
					}
				};

				walk(document.body);
			}

			// highlights text on webpage
			if (message.type === 'HIGHLIGHT_TEXT') {
				const { target } = message;
				if (!target) return;

				// remove previous highlights
				const existing = document.querySelectorAll('.highlight');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});

				let firstMatch: HTMLElement | null = null;

				const walk = (node: Node) => {
					if (node.nodeType === Node.TEXT_NODE) {
						const parent = node.parentNode;
						if (!parent) return;

						let text = node.nodeValue || '';
						const frag = document.createDocumentFragment();
						let cursor = 0;

						while (cursor < text.length) {
							const normalizedText = normalizeSpaces(text.slice(cursor));
							const normalizedTarget = normalizeSpaces(target);

							const index = normalizedText.indexOf(normalizedTarget);
							if (index === -1) {
								frag.append(document.createTextNode(text.slice(cursor)));
								break;
							}

							if (index > 0) frag.append(document.createTextNode(text.slice(cursor, cursor + index)));

							const matchedText = text.slice(cursor + index, cursor + index + normalizedTarget.length);

							// css for highlighted text
							const span = document.createElement('span');
							span.className = 'highlight';
							span.style.backgroundColor = 'rgb(254, 240, 138)';
							span.textContent = matchedText;
							frag.append(span);

							if (!firstMatch) firstMatch = span;
							cursor = cursor + index + normalizedTarget.length;
						}

						parent.replaceChild(frag, node);
					} else if (node.nodeType === Node.ELEMENT_NODE) {
						const tag = (node as HTMLElement).tagName;
						if (tag === 'SCRIPT' || tag === 'STYLE') return;
						Array.from(node.childNodes).forEach(walk);
					}
				};

				walk(document.body);

				// scroll to the corresponding sentence on webpage
				if (firstMatch) {
					(firstMatch as HTMLElement).scrollIntoView({ behavior: 'smooth', block: 'center' });
				}
			}

			// clears highlights from webpage
			if (message.type === 'CLEAR_HIGHLIGHTS') {
				const existing = document.querySelectorAll('.highlight');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});
			}
			// clears underlines from webpage
			if (message.type === 'CLEAR_UNDERLINES') {
				const existing = document.querySelectorAll('.my-extension-underline');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});
			}
			// clear selected text from webpage
			if (message.type === 'CLEAR_SELECTION') {
				window.getSelection()?.removeAllRanges();
			}
		});
	},
});

function normalizeSpaces(str: string) {
	return str.replace(/\u00A0/g, ' ');
}
