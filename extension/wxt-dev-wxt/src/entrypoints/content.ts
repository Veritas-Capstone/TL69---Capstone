export default defineContentScript({
	matches: ['<all_urls>'],
	main(ctx) {
		browser.runtime.onMessage.addListener(async (message) => {
			// selects text from entire webpage
			// called when "scan current page" popup button clicked
			if (message.type === 'GET_PAGE_TEXT') {
				await browser.storage.local.set({ selectedText: document.body.innerText });
			}

			// underlines valid sentences (targets) on webpage
			if (message.type === 'UNDERLINE_SELECTION_VALID') {
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
								// Replace &nbsp; in both text and target with normal space for matching
								const normalizedText = text.slice(cursor).replace(/\u00A0/g, ' ');
								const normalizedTarget = target.replace(/\u00A0/g, ' ');

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
							span.style.textDecorationColor = 'rgb(74, 222, 128)';
							span.style.textDecorationThickness = '2px';
							span.textContent = matchedTarget;
							span.style.cursor = 'pointer';

							span.addEventListener('click', (e) => {
								e.stopPropagation();
								browser.runtime.sendMessage({
									type: 'UNDERLINE_CLICKED',
									text: matchedTarget,
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
			// underlines invalid sentences (targets) on webpage
			if (message.type === 'UNDERLINE_SELECTION_INVALID') {
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
								// Replace &nbsp; in both text and target with normal space for matching
								const normalizedText = text.slice(cursor).replace(/\u00A0/g, ' ');
								const normalizedTarget = target.replace(/\u00A0/g, ' ');

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
							span.style.textDecorationColor = 'red';
							span.style.textDecorationThickness = '2px';
							span.textContent = matchedTarget;
							span.style.cursor = 'pointer';

							span.addEventListener('click', (e) => {
								e.stopPropagation();
								browser.runtime.sendMessage({
									type: 'UNDERLINE_CLICKED',
									text: matchedTarget,
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
			// used when hovering over claims in popup to see
			// which claim corresponds to which sentences
			if (message.type === 'HIGHLIGHT_TEXT') {
				const { target } = message;
				if (!target) return;

				// Remove previous highlights
				const existing = document.querySelectorAll('.my-extension-highlight');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});

				const walk = (node: Node) => {
					if (node.nodeType === Node.TEXT_NODE) {
						const parent = node.parentNode;
						if (!parent) return;

						let text = node.nodeValue || '';
						const frag = document.createDocumentFragment();
						let cursor = 0;

						while (cursor < text.length) {
							// Normalize spaces for matching
							const normalizedText = text.slice(cursor).replace(/\u00A0/g, ' ');
							const normalizedTarget = target.replace(/\u00A0/g, ' ');

							const index = normalizedText.indexOf(normalizedTarget);
							if (index === -1) {
								frag.append(document.createTextNode(text.slice(cursor)));
								break;
							}

							// Append text before the match
							if (index > 0) frag.append(document.createTextNode(text.slice(cursor, cursor + index)));

							// Matched text preserves original spacing
							const matchedText = text.slice(cursor + index, cursor + index + normalizedTarget.length);

							const span = document.createElement('span');
							span.className = 'my-extension-highlight';
							span.style.backgroundColor = 'rgb(254, 240, 138)';
							span.textContent = matchedText;
							frag.append(span);

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
			}

			// clears above highlights
			if (message.type === 'CLEAR_HIGHLIGHTS') {
				const existing = document.querySelectorAll('.my-extension-highlight');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});
			}
			// clears above underlines
			if (message.type === 'CLEAR_UNDERLINES') {
				const existing = document.querySelectorAll('.my-extension-underline');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});
			}
		});
	},
});
