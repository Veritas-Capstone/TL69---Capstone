/* don't look at this, chatGPT wrote it all and I do not know what it does 🔥🔥🔥 */
export function underlineSentences(
	sentences: { text?: string; category?: string; claim?: string; label?: string; valid: boolean }[],
) {
	console.log(sentences);
	const normalize = (str: string) =>
		str.replace(/\s+/g, ' ').replace(/[“”]/g, '"').replace(/[‘’]/g, "'").trim();

	function mapNormalizedIndexToRaw(raw: string, normalizedIndex: number, inclusive = false) {
		let rawIdx = 0;
		let normIdx = 0;

		while (rawIdx < raw.length) {
			const char = raw[rawIdx];
			const normalizedChar =
				char === '“' || char === '”'
					? '"'
					: char === '‘' || char === '’'
						? "'"
						: /\s/.test(char)
							? ' '
							: char;

			if (normalizedChar === ' ') {
				if (normIdx === normalizedIndex) break;
				normIdx++;
				rawIdx++;
				while (rawIdx < raw.length && /\s/.test(raw[rawIdx])) rawIdx++;
				continue;
			}

			if (normIdx === normalizedIndex) break;

			normIdx++;
			rawIdx++;
		}

		if (inclusive && rawIdx < raw.length) rawIdx++;
		return rawIdx;
	}

	function unwrapInlineTags(root: ParentNode) {
		const nodes = root.querySelectorAll('a, em, strong, cite, span');
		nodes.forEach((node) => {
			if (node.closest('nav')) return;
			if (node.closest('div')) return;
			const parent = node.parentNode;
			if (!parent) return;
			while (node.firstChild) parent.insertBefore(node.firstChild, node);
			parent.removeChild(node);
		});
	}

	const containers = Array.from(
		document.querySelectorAll<HTMLElement>(
			'p, h1, h2, h3, h4, h5, h6, li, span, figcaption, blockquote, footer, aside, cite',
		),
	).filter((el) => !el.closest('script, style'));

	containers.forEach(unwrapInlineTags);

	const textElements = containers.filter((el) =>
		Array.from(el.childNodes).some((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim()),
	);

	const elementTexts = textElements.map((el) => el.textContent || '');
	const flatText = elementTexts.join('\n');
	const normalizedFlat = normalize(flatText);

	// 🔴 NEW: track missing indices
	const missingIndices: number[] = [];

	const matches = sentences
		.map((s, idx) => {
			const sentenceText = s.text ?? s.claim;
			if (!sentenceText) {
				missingIndices.push(idx);
				return null;
			}

			const normalizedSentence = normalize(sentenceText);
			const start = normalizedFlat.indexOf(normalizedSentence);

			if (start === -1) {
				missingIndices.push(idx);
				return null;
			}

			return {
				idx,
				valid: s.valid,
				start,
				end: start + normalizedSentence.length,
			};
		})
		.filter(Boolean) as { idx: number; valid: boolean; start: number; end: number }[];

	let flatCursor = 0;

	textElements.forEach((el) => {
		const elText = el.textContent || '';
		const normalizedElText = normalize(elText);
		const elStart = flatCursor;
		const elEnd = flatCursor + normalizedElText.length;

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
			const rawCursor = mapNormalizedIndexToRaw(elText, cursor);
			const rawLocalStart = mapNormalizedIndexToRaw(elText, localStart);

			if (rawLocalStart > rawCursor) {
				fragment.appendChild(document.createTextNode(elText.slice(rawCursor, rawLocalStart)));
			}

			const rawEnd = mapNormalizedIndexToRaw(elText, localEnd, true);

			const span = document.createElement('span');
			span.className = `underline-${idx}`;
			span.style.cursor = 'pointer';
			span.style.paddingTop = '4px';
			span.style.paddingBottom = '4px';
			span.style.textDecoration = 'underline';
			span.style.backgroundClip = 'content-box';
			if (sentences[idx].category) {
				span.style.textDecorationColor =
					sentences[idx].category === 'Left-leaning'
						? '#60a5fa'
						: sentences[idx].category === 'Right-leaning'
							? '#f87171'
							: '#c084fc';
			} else if (sentences[idx].label) {
				span.style.textDecorationColor =
					sentences[idx].label === 'SUPPORTED'
						? '#4ade80'
						: sentences[idx].label === 'REFUTED'
							? '#f87171'
							: '#9ca3af';
			} else {
				span.style.textDecorationColor = valid ? 'rgba(74, 222, 128, 0.8)' : 'rgba(244, 63, 94, 0.8)';
			}
			span.textContent = elText.slice(rawLocalStart, rawEnd);

			span.addEventListener('mouseenter', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx });
			});
			span.addEventListener('mouseleave', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx: undefined });
			});
			span.addEventListener('click', (e) => {
				e.stopPropagation();

				browser.runtime.sendMessage({
					type: 'OPEN_SIDEBAR',
				});
			});

			fragment.appendChild(span);
			cursor = localEnd;
		});

		if (cursor < normalizedElText.length) {
			const rawCursor = mapNormalizedIndexToRaw(elText, cursor);
			fragment.appendChild(document.createTextNode(elText.slice(rawCursor)));
		}

		el.textContent = '';
		el.appendChild(fragment);

		flatCursor += normalizedElText.length + 1;
	});

	return missingIndices;
}
