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
				if (!targets || !Array.isArray(targets) || targets.length === 0) return;

				console.log('HIGHLIGHT_TEXT message received:', targets);

				// 1) Remove previous highlights

				// 2) Collect all text nodes once
				const textNodes: { node: Text; parent: Node; startOffset: number; endOffset: number }[] = [];
				let cumulativeOffset = 0;

				const isBlockElement = (tag: string) =>
					['P', 'DIV', 'LI', 'SECTION', 'ARTICLE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(tag);

				const collect = (n: Node) => {
					if (n.nodeType === Node.TEXT_NODE) {
						const text = n.nodeValue || '';
						textNodes.push({
							node: n as Text,
							parent: n.parentNode!,
							startOffset: cumulativeOffset,
							endOffset: cumulativeOffset + text.length,
						});
						cumulativeOffset += text.length;
						if (isBlockElement((n.parentNode as HTMLElement).tagName)) cumulativeOffset += 1;
					} else if (n.nodeType === Node.ELEMENT_NODE) {
						const tag = (n as HTMLElement).tagName;
						if (!['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(tag)) {
							Array.from(n.childNodes).forEach(collect);
						}
					}
				};
				collect(document.body);

				// 3) Concatenate all text
				let fullText = '';
				textNodes.forEach(({ node, parent }) => {
					fullText += node.nodeValue || '';
					if (isBlockElement((parent as HTMLElement).tagName)) fullText += '\n';
				});

				let firstMatchEl: HTMLElement | null = null;

				// 4) Loop through all targets
				targets.forEach((target) => {
					const normalizedTarget = normalizeSpaces(target);
					if (!normalizedTarget) return;

					const escaped = normalizedTarget.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
					const pattern = escaped.replace(/ +/g, '\\s+');
					const re = new RegExp(pattern, 'i');

					const match = re.exec(fullText);
					if (!match || typeof match.index !== 'number') return;

					const matchStart = match.index;
					const matchEnd = match.index + match[0].length;

					// 5) Highlight across nodes
					for (const { node, parent, startOffset, endOffset } of textNodes) {
						if (endOffset <= matchStart || startOffset >= matchEnd) continue;

						const nodeStart = Math.max(0, matchStart - startOffset);
						const nodeEnd = Math.min(node.nodeValue!.length, matchEnd - startOffset);

						if (nodeStart >= nodeEnd) continue;

						// Split text node and wrap matched part
						const raw = node.nodeValue || '';
						const beforeText = raw.slice(0, nodeStart);
						const matchedText = raw.slice(nodeStart, nodeEnd);
						const afterText = raw.slice(nodeEnd);

						const frag = document.createDocumentFragment();
						if (beforeText) frag.appendChild(document.createTextNode(beforeText));

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
								text: normalizeSpacesForPattern(span.textContent || ''),
							});
						});
						span.addEventListener('mouseleave', () => {
							browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', text: '' });
						});

						frag.appendChild(span);

						if (afterText) frag.appendChild(document.createTextNode(afterText));

						if (parent.contains(node)) parent.replaceChild(frag, node);

						if (!firstMatchEl) firstMatchEl = span;
					}
				});
			}

			if (message.type === 'HIGHLIGHT_TEXT') {
				const { target } = message;
				if (!target) return;

				console.log('HIGHLIGHT_TEXT message received:', target);

				// Remove previous highlights
				document.querySelectorAll('.highlight').forEach((el) => {
					const p = el.parentNode;
					if (p) {
						console.log('Removing previous highlight:', el.textContent);
						p.replaceChild(document.createTextNode(el.textContent || ''), el);
					}
				});

				const normalizedTarget = normalizeSpaces(target);
				if (!normalizedTarget) return;
				console.log('Normalized target:', normalizedTarget);

				// 1) Collect all text nodes
				const textNodes: { node: Text; parent: Node }[] = [];
				const collect = (n: Node) => {
					if (n.nodeType === Node.TEXT_NODE) {
						textNodes.push({ node: n as Text, parent: n.parentNode! });
					} else if (n.nodeType === Node.ELEMENT_NODE) {
						const tag = (n as HTMLElement).tagName;
						if (!['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(tag)) {
							Array.from(n.childNodes).forEach(collect);
						}
					}
				};
				collect(document.body);
				console.log('Collected text nodes count:', textNodes.length);

				const nodeRawStrings = textNodes.map(({ node }) => node.nodeValue || '');
				const rawConcat = nodeRawStrings.join(' ');
				console.log('Concatenated text:', rawConcat.slice(0, 500), '...');

				// 2) Regex for matching ignoring multiple spaces
				const escaped = normalizedTarget.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
				const pattern = escaped.replace(/ +/g, '\\s+');
				const re = new RegExp(pattern, 'i');
				console.log('Regex pattern:', re);

				const match = re.exec(rawConcat);
				if (!match || typeof match.index !== 'number') {
					console.log('Target not found in page text');
					document.querySelectorAll('.highlight').forEach((el) => {
						const p = el.parentNode;
						if (p && p.contains(el)) {
							console.log('Removing previous highlight:', el.textContent);
							p.replaceChild(document.createTextNode(el.textContent || ''), el);
						}
					});

					const normalizedTarget = normalizeSpaces(target);
					if (!normalizedTarget) return;
					console.log('Normalized target:', normalizedTarget);

					// Collect all text nodes
					const textNodes: { node: Text; parent: Node; startOffset: number; endOffset: number }[] = [];
					let cumulativeOffset = 0;

					const isBlockElement = (tag: string) =>
						['P', 'DIV', 'LI', 'SECTION', 'ARTICLE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(tag);

					const collect = (n: Node) => {
						if (n.nodeType === Node.TEXT_NODE) {
							const text = n.nodeValue || '';
							textNodes.push({
								node: n as Text,
								parent: n.parentNode!,
								startOffset: cumulativeOffset,
								endOffset: cumulativeOffset + text.length,
							});
							cumulativeOffset += text.length;
							// Add 1 for block boundary if parent is block element
							if (isBlockElement((n.parentNode as HTMLElement).tagName)) cumulativeOffset += 1;
						} else if (n.nodeType === Node.ELEMENT_NODE) {
							const tag = (n as HTMLElement).tagName;
							if (!['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(tag)) {
								Array.from(n.childNodes).forEach(collect);
							}
						}
					};
					collect(document.body);
					console.log('Collected text nodes:', textNodes.length);

					// Concatenate raw text WITHOUT extra spaces, but keep \n for blocks
					let fullText = '';
					textNodes.forEach(({ node, parent }) => {
						fullText += node.nodeValue || '';
						if (isBlockElement((parent as HTMLElement).tagName)) fullText += '\n';
					});
					console.log('Full text length:', fullText.length);

					// Regex for matching normalized target text ignoring multiple spaces
					const escaped = normalizedTarget.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
					const pattern = escaped.replace(/ +/g, '\\s+');
					const re = new RegExp(pattern, 'i');
					const match = re.exec(fullText);

					if (!match || typeof match.index !== 'number') {
						console.log('Target not found in page text');
						return;
					}

					const matchStart = match.index;
					const matchEnd = match.index + match[0].length;
					console.log('Match found:', match[0]);
					console.log('Match offsets:', matchStart, matchEnd);

					// Highlight matched text across nodes
					let firstMatchEl: HTMLElement | null = null;
					for (const { node, parent, startOffset, endOffset } of textNodes) {
						if (endOffset <= matchStart || startOffset >= matchEnd) continue;

						const nodeStart = Math.max(0, matchStart - startOffset);
						const nodeEnd = Math.min(node.nodeValue!.length, matchEnd - startOffset);

						const frag = document.createDocumentFragment();
						const raw = node.nodeValue || '';

						if (nodeStart > 0) frag.appendChild(document.createTextNode(raw.slice(0, nodeStart)));

						const span = document.createElement('span');
						span.className = 'highlight';
						span.style.backgroundColor = 'rgb(254, 240, 138)';
						span.textContent = raw.slice(nodeStart, nodeEnd);
						frag.appendChild(span);

						if (!firstMatchEl) firstMatchEl = span;

						if (nodeEnd < raw.length) frag.appendChild(document.createTextNode(raw.slice(nodeEnd)));

						try {
							if (parent.contains(node)) parent.replaceChild(frag, node);
							console.log('Replaced node with highlighted span');
						} catch (e) {
							console.error('Error replacing node:', e, node, parent);
						}
					}

					if (firstMatchEl) {
						try {
							firstMatchEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
						} catch (e) {
							console.error('Error scrolling to first match:', e);
						}
					}
					return;
				}

				const matchStart = match.index;
				const matchEnd = match.index + match[0].length;
				console.log('Match found:', match[0]);
				console.log('Match offsets:', matchStart, matchEnd);

				// 3) Compute cumulative offsets
				const cumulative: number[] = [];
				let acc = 0;
				for (let i = 0; i < nodeRawStrings.length; i++) {
					cumulative.push(acc);
					acc += nodeRawStrings[i].length + 1; // +1 for space separator
				}
				console.log('Cumulative offsets:', cumulative.slice(0, 20));

				const findNodeForOffset = (offset: number) => {
					for (let i = 0; i < nodeRawStrings.length; i++) {
						const start = cumulative[i];
						const end = start + nodeRawStrings[i].length;
						if (offset >= start && offset < end) return { nodeIndex: i, localOffset: offset - start };
					}
					const last = nodeRawStrings.length - 1;
					return { nodeIndex: last, localOffset: nodeRawStrings[last].length };
				};

				const startMap = findNodeForOffset(matchStart);
				const endMap = findNodeForOffset(Math.max(matchEnd - 1, matchStart));
				console.log('Start node mapping:', startMap, 'End node mapping:', endMap);

				// 4) Replace text nodes with span
				let firstMatchEl: HTMLElement | null = null;
				for (let ni = startMap.nodeIndex; ni <= endMap.nodeIndex; ni++) {
					const { node, parent } = textNodes[ni];
					const raw = node.nodeValue || '';
					const nodeStartOffset = ni === startMap.nodeIndex ? startMap.localOffset : 0;
					const nodeEndOffset = ni === endMap.nodeIndex ? endMap.localOffset + 1 : raw.length;

					console.log(`Processing node ${ni}: "${raw}"`, nodeStartOffset, nodeEndOffset);

					if (nodeStartOffset >= nodeEndOffset) continue;

					const frag = document.createDocumentFragment();
					if (nodeStartOffset > 0) frag.appendChild(document.createTextNode(raw.slice(0, nodeStartOffset)));

					const span = document.createElement('span');
					span.className = 'highlight';
					span.style.backgroundColor = 'rgb(254, 240, 138)';
					span.textContent = raw.slice(nodeStartOffset, nodeEndOffset);
					frag.appendChild(span);

					if (!firstMatchEl) firstMatchEl = span;

					if (nodeEndOffset < raw.length) frag.appendChild(document.createTextNode(raw.slice(nodeEndOffset)));

					try {
						parent.replaceChild(frag, node);
						console.log('Replaced node with highlighted span');
					} catch (e) {
						console.error('Error replacing node:', e, node, parent);
					}
				}

				if (firstMatchEl) {
					try {
						firstMatchEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
					} catch (e) {
						console.error('Error scrolling to first match:', e);
					}
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

// highlights text on webpage
// helper: escape regex special chars
function escapeForRegex(s: string) {
	return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// helper: collapse multiple whitespace in target to single spaces for pattern building
function normalizeSpacesForPattern(s: string) {
	return s
		.replace(/\u00A0/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();
}
