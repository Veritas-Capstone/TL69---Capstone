export function underlineSentences(
	sentences: { text?: string; category?: string; claim?: string; label?: string; valid: boolean }[],
) {
	const normalize = (str: string) => str.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");

	const textNodes: Text[] = [];
	const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
		acceptNode: (node) => {
			const parent = node.parentElement;
			if (!parent || parent.closest('script, style, noscript, nav, header, footer')) {
				return NodeFilter.FILTER_REJECT;
			}
			return NodeFilter.FILTER_ACCEPT;
		},
	});

	let currentNode: Node | null;
	while ((currentNode = walker.nextNode())) {
		textNodes.push(currentNode as Text);
	}

	let fullRawText = '';
	const charToNodeMap: { node: Text; offset: number }[] = [];

	textNodes.forEach((node) => {
		const text = node.nodeValue || '';
		for (let i = 0; i < text.length; i++) {
			charToNodeMap.push({ node, offset: i });
		}
		fullRawText += text;
	});

	let strippedText = '';
	const strippedToRawIndex: number[] = [];

	for (let i = 0; i < fullRawText.length; i++) {
		const char = fullRawText[i];
		const normChar = normalize(char);
		if (!/\s/.test(normChar)) {
			strippedToRawIndex.push(i);
			strippedText += normChar;
		}
	}

	const missingIndices: number[] = [];
	const nodeHighlights = new Map<Text, { start: number; end: number; matchIdx: number; config: any }[]>();

	sentences.forEach((s, idx) => {
		const sentenceText = s.text ?? s.claim;
		if (!sentenceText) {
			missingIndices.push(idx);
			return;
		}

		const target = normalize(sentenceText).replace(/\s+/g, '');
		const startStripped = strippedText.indexOf(target);

		if (startStripped === -1) {
			missingIndices.push(idx);
			return;
		}

		const endStripped = startStripped + target.length;

		const rawStart = strippedToRawIndex[startStripped];
		const rawEnd = strippedToRawIndex[endStripped - 1] + 1;

		for (let i = rawStart; i < rawEnd; i++) {
			const mapping = charToNodeMap[i];
			if (mapping) {
				const existing = nodeHighlights.get(mapping.node) || [];
				const last = existing[existing.length - 1];

				if (last && last.matchIdx === idx && last.end === mapping.offset) {
					last.end = mapping.offset + 1;
				} else {
					existing.push({
						start: mapping.offset,
						end: mapping.offset + 1,
						matchIdx: idx,
						config: s,
					});
				}
				nodeHighlights.set(mapping.node, existing);
			}
		}
	});

	nodeHighlights.forEach((highlights, node) => {
		const parent = node.parentNode;
		if (!parent) return;

		highlights.sort((a, b) => b.start - a.start);

		const fullText = node.nodeValue || '';
		let lastIdx = fullText.length;
		const fragment = document.createDocumentFragment();

		highlights.forEach((h) => {
			if (lastIdx > h.end) {
				fragment.prepend(document.createTextNode(fullText.slice(h.end, lastIdx)));
			}

			const span = document.createElement('span');
			span.className = `underline-${h.matchIdx}`;
			span.style.textDecoration = 'underline';
			span.style.backgroundClip = 'content-box';
			span.style.cursor = 'pointer';
			span.style.padding = '2px 0';

			if (h.config.category) {
				span.style.textDecorationColor =
					h.config.category === 'Left-leaning'
						? '#60a5fa'
						: h.config.category === 'Right-leaning'
							? '#f87171'
							: '#c084fc';
			} else if (h.config.label) {
				span.style.textDecorationColor =
					h.config.label === 'SUPPORTED' ? '#4ade80' : h.config.label === 'REFUTED' ? '#f87171' : '#9ca3af';
			} else {
				span.style.textDecorationColor = h.config.valid
					? 'rgba(74, 222, 128, 0.8)'
					: 'rgba(244, 63, 94, 0.8)';
			}

			span.textContent = fullText.slice(h.start, h.end);

			span.addEventListener('mouseenter', () =>
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx: h.matchIdx }),
			);
			span.addEventListener('mouseleave', () =>
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx: undefined }),
			);
			span.addEventListener('click', (e) => {
				e.stopPropagation();
				browser.runtime.sendMessage({ type: 'OPEN_SIDEBAR' });
			});

			fragment.prepend(span);
			lastIdx = h.start;
		});

		if (lastIdx > 0) {
			fragment.prepend(document.createTextNode(fullText.slice(0, lastIdx)));
		}

		parent.replaceChild(fragment, node);
	});

	return missingIndices;
}
