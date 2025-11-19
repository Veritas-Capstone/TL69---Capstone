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

				const normalizedTargets = targets.map((t) => normalizeSpaces(t));
				let firstMatch: HTMLElement | null = null;

				// Collect all text nodes on the page
				const textNodes: { node: Text; parent: Node }[] = [];
				const collectTextNodes = (node: Node) => {
					if (node.nodeType === Node.TEXT_NODE) {
						textNodes.push({ node: node as Text, parent: node.parentNode! });
					} else if (node.nodeType === Node.ELEMENT_NODE) {
						const tag = (node as HTMLElement).tagName;
						if (tag !== 'SCRIPT' && tag !== 'STYLE') {
							Array.from(node.childNodes).forEach(collectTextNodes);
						}
					}
				};
				collectTextNodes(document.body);

				// Concatenate normalized text of all nodes with spaces
				const fullText = textNodes.map(({ node }) => normalizeSpaces(node.nodeValue || '')).join(' ');

				normalizedTargets.forEach((target) => {
					const startIndex = fullText.indexOf(target);
					if (startIndex === -1) return; // target not found

					const targetEnd = startIndex + target.length;
					let charCount = 0;

					for (const { node, parent } of textNodes) {
						const nodeText = normalizeSpaces(node.nodeValue || '');
						const nodeStart = charCount;
						const nodeEnd = charCount + nodeText.length;

						if (nodeEnd > startIndex && nodeStart < targetEnd) {
							const frag = document.createDocumentFragment();
							const originalText = node.nodeValue || '';

							const overlapStart = Math.max(0, startIndex - nodeStart);
							const overlapEnd = Math.min(nodeText.length, targetEnd - nodeStart);

							if (overlapStart > 0) {
								frag.append(document.createTextNode(originalText.slice(0, overlapStart)));
							}

							const matchedText = originalText.slice(overlapStart, overlapEnd);
							const span = document.createElement('span');
							span.className = 'my-extension-underline';
							span.style.textDecoration = 'underline';
							span.style.textDecorationColor = message.valid ? 'rgb(74, 222, 128)' : 'red';
							span.style.textDecorationThickness = '2px';
							span.style.paddingTop = '4px';
							span.style.paddingBottom = '4px';
							span.style.lineHeight = '2px';
							span.style.cursor = 'pointer';
							span.textContent = matchedText;

							span.addEventListener('mouseenter', () => {
								browser.runtime.sendMessage({
									type: 'UNDERLINE_HOVER',
									text: normalizeSpaces(matchedText),
								});
							});

							span.addEventListener('mouseleave', () => {
								browser.runtime.sendMessage({
									type: 'UNDERLINE_HOVER',
									text: '',
								});
							});

							frag.append(span);

							if (!firstMatch) firstMatch = span;

							if (overlapEnd < originalText.length) {
								frag.append(document.createTextNode(originalText.slice(overlapEnd)));
							}

							parent.replaceChild(frag, node);
						}

						charCount += nodeText.length + 1; // +1 for space between nodes
					}
				});
			}

			// highlights text on webpage
			if (message.type === 'HIGHLIGHT_TEXT') {
				const { target } = message;
				if (!target) return;

				// Remove previous highlights
				const existing = document.querySelectorAll('.highlight');
				existing.forEach((span) => {
					const parent = span.parentNode;
					if (parent) parent.replaceChild(document.createTextNode(span.textContent || ''), span);
				});

				const normalizedTarget = normalizeSpaces(target);
				let firstMatch: HTMLElement | null = null;

				// Flatten all text nodes on the page into an array with their parent references
				const textNodes: { node: Text; parent: Node }[] = [];
				const collectTextNodes = (node: Node) => {
					if (node.nodeType === Node.TEXT_NODE) {
						textNodes.push({ node: node as Text, parent: node.parentNode! });
					} else if (node.nodeType === Node.ELEMENT_NODE) {
						const tag = (node as HTMLElement).tagName;
						if (tag !== 'SCRIPT' && tag !== 'STYLE') {
							Array.from(node.childNodes).forEach(collectTextNodes);
						}
					}
				};
				collectTextNodes(document.body);

				// Concatenate all normalized text from nodes
				const fullText = textNodes.map(({ node }) => normalizeSpaces(node.nodeValue || '')).join(' ');

				// Find the start index of target in fullText
				const startIndex = fullText.indexOf(normalizedTarget);
				if (startIndex === -1) return; // target not found

				// Walk through nodes and wrap the matching portion
				let charCount = 0;
				let targetStart = startIndex;
				let targetEnd = startIndex + normalizedTarget.length;

				for (const { node, parent } of textNodes) {
					const nodeText = normalizeSpaces(node.nodeValue || '');
					const nodeStart = charCount;
					const nodeEnd = charCount + nodeText.length;

					// Check if this node overlaps the target
					if (nodeEnd > targetStart && nodeStart < targetEnd) {
						const frag = document.createDocumentFragment();
						const nodeTextOriginal = node.nodeValue || '';

						let localCursor = 0;
						const overlapStart = Math.max(0, targetStart - nodeStart);
						const overlapEnd = Math.min(nodeText.length, targetEnd - nodeStart);

						if (overlapStart > 0) {
							frag.append(document.createTextNode(nodeTextOriginal.slice(0, overlapStart)));
						}

						const matchedText = nodeTextOriginal.slice(overlapStart, overlapEnd);
						const span = document.createElement('span');
						span.className = 'highlight';
						span.style.backgroundColor = 'rgb(254, 240, 138)';
						span.textContent = matchedText;
						frag.append(span);

						if (!firstMatch) firstMatch = span;

						if (overlapEnd < nodeText.length) {
							frag.append(document.createTextNode(nodeTextOriginal.slice(overlapEnd)));
						}

						parent.replaceChild(frag, node);
					}

					charCount += nodeText.length + 1; // +1 for space between nodes in concatenated string
				}

				// Scroll to the first matched element
				if (firstMatch) {
					firstMatch.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
