export default defineContentScript({
	matches: ['<all_urls>'],
	main() {
		browser.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
			// selects text from entire webpage
			if (message.type === 'GET_PAGE_TEXT') {
				await browser.storage.local.set({ selectedText: document.body.innerText });
			}

			if (message.type === 'GET_SELECTION_PARAGRAPHS') {
				const selection = window.getSelection();
				if (!selection || selection.rangeCount === 0) {
					return Promise.resolve({ paragraphs: [] });
				}

				const range = selection.getRangeAt(0);

				const container = document.createElement('div');
				container.appendChild(range.cloneContents());

				const paragraphs: string[] = [];

				container.querySelectorAll('p, div, li').forEach((el) => {
					const text = el.textContent?.trim();
					if (text) paragraphs.push(text);
				});

				if (paragraphs.length === 0) {
					const text = container.textContent?.trim();
					if (text) paragraphs.push(text);
				}

				console.log('Extracted paragraphs:', paragraphs);
				sendResponse(paragraphs);
				return true;
			}

			// underlines sentences on webpage
			if (message.type === 'UNDERLINE_SELECTION') {
				const { sentences } = message;
				console.log(sentences);
				underlineSentences(sentences);

				/*
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
						span.style.textDecorationColor = message.valid
							? 'rgba(74, 222, 128, 0.8)'
							: 'rgba(244, 63, 94, 0.8)';
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
				*/
			}

			if (message.type === 'HIGHLIGHT_TEXT') {
				const { idx } = message;
				const spans = document.querySelectorAll(`span.underline-${idx}`);

				spans.forEach((span) => {
					// Add yellow background
					(span as HTMLElement).style.backgroundColor = 'yellow';
				});

				const firstSpan = spans[0] as HTMLElement | undefined;
				if (firstSpan) {
					firstSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
				}

				/*
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
						span.style.backgroundColor = message.valid ? 'rgba(74, 222, 128, 0.4)' : 'rgba(244, 63, 94, 0.4)';
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
					span.style.backgroundColor = message.valid ? 'rgba(74, 222, 128, 0.4)' : 'rgba(244, 63, 94, 0.4)';
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
				*/
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

					// Replace the <span> with its text content
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

function underlineSentences(sentences: { text: string; valid: boolean }[]) {
	const normalize = (str: string) =>
		str.replace(/\s+/g, ' ').replace(/[“”]/g, '"').replace(/[‘’]/g, "'").trim();

	// Select text elements, skip figures/media
	const allElements = Array.from(
		document.querySelectorAll<HTMLElement>('p, h1, h2, h3, h4, h5, h6, div, li, span')
	);

	const textElements = allElements.filter((el) => {
		// Only include elements with at least one direct text node
		return Array.from(el.childNodes).some((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim());
	});

	// Flatten text
	const elementTexts = textElements.map((el) => el.textContent || '');
	const flatText = elementTexts.join('\n');
	const normalizedFlat = normalize(flatText);

	// Precompute sentence matches in flat text
	const matches = sentences
		.map((s, idx) => {
			const start = normalizedFlat.indexOf(normalize(s.text));
			if (start === -1) return null;
			return {
				idx,
				valid: s.valid,
				start,
				end: start + normalize(s.text).length,
			};
		})
		.filter(Boolean) as {
		idx: number;
		valid: boolean;
		start: number;
		end: number;
	}[];

	let flatCursor = 0;

	textElements.forEach((el) => {
		const elText = el.textContent || '';
		const normalizedElText = normalize(elText);
		const elStart = flatCursor;
		const elEnd = flatCursor + normalizedElText.length;

		// Matches overlapping this element
		const elMatches = matches
			.filter((m) => m.start < elEnd && m.end > elStart)
			.map((m) => ({
				...m,
				localStart: Math.max(0, m.start - elStart),
				localEnd: Math.min(normalizedElText.length, m.end - elStart),
			}))
			.sort((a, b) => a.localStart - b.localStart);

		if (!elMatches.length) {
			flatCursor += normalizedElText.length + 1;
			return;
		}

		const fragment = document.createDocumentFragment();
		let cursor = 0;

		elMatches.forEach(({ localStart, localEnd, idx, valid }) => {
			if (cursor < localStart) {
				fragment.appendChild(document.createTextNode(elText.slice(cursor, localStart)));
			}

			const span = document.createElement('span');
			span.className = `underline-${idx}`;
			span.style.cursor = 'pointer';
			span.style.textDecoration = 'underline';
			span.style.textDecorationColor = valid ? 'rgba(74, 222, 128, 0.8)' : 'rgba(244, 63, 94, 0.8)';
			span.dataset.underlined = 'true';
			span.textContent = elText.slice(localStart, localEnd);

			span.addEventListener('mouseenter', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx });
			});
			span.addEventListener('mouseleave', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx: undefined });
			});

			fragment.appendChild(span);
			cursor = localEnd;
		});

		if (cursor < elText.length) {
			fragment.appendChild(document.createTextNode(elText.slice(cursor)));
		}

		el.textContent = '';
		el.appendChild(fragment);

		flatCursor += normalizedElText.length + 1;
	});
}
