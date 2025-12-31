export function underlineSentences(sentences: { text?: string; claim?: string; valid: boolean }[]) {
	const normalize = (str: string) =>
		str.replace(/\s+/g, ' ').replace(/[“”]/g, '"').replace(/[‘’]/g, "'").trim();

	/**
	 * Map normalized index to raw text index safely
	 */
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

	// Select all text elements, skip script/style/figure/blockquote if desired
	const allElements = Array.from(
		document.querySelectorAll<HTMLElement>(
			'p, h1, h2, h3, h4, h5, h6, li, div, span, figcaption, strong, em, blockquote, footer, aside'
		)
	);

	const textElements = allElements.filter((el) => {
		// Skip elements inside script/style
		if (el.closest('script, style')) return false;

		// Only elements with at least one direct text node
		return Array.from(el.childNodes).some((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim());
	});

	// Flatten text
	const elementTexts = textElements.map((el) => el.textContent || '');
	const flatText = elementTexts.join('\n');
	const normalizedFlat = normalize(flatText);

	// Precompute matches
	const matches = sentences
		.map((s, idx) => {
			const sentenceText = s.text ?? s.claim;
			if (!sentenceText) return null;
			const normalizedSentence = normalize(sentenceText);
			const start = normalizedFlat.indexOf(normalizedSentence);
			if (start === -1) return null;
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

			// Add text before the span
			if (rawLocalStart > rawCursor) {
				fragment.appendChild(document.createTextNode(elText.slice(rawCursor, rawLocalStart)));
			}

			// Add the span
			const rawStart = rawLocalStart;
			const rawEnd = mapNormalizedIndexToRaw(elText, localEnd, true);

			const span = document.createElement('span');
			span.className = `underline-${idx}`;
			span.style.cursor = 'pointer';
			span.style.paddingTop = '2px';
			span.style.paddingBottom = '2px';
			span.style.textDecoration = 'underline';
			span.style.textDecorationColor = valid ? 'rgba(74, 222, 128, 0.8)' : 'rgba(244, 63, 94, 0.8)';
			span.dataset.underlined = 'true';
			span.textContent = elText.slice(rawStart, rawEnd);

			span.addEventListener('mouseenter', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx });
			});
			span.addEventListener('mouseleave', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx: undefined });
			});

			fragment.appendChild(span);
			cursor = localEnd;
		});

		// Remaining text after last span
		if (cursor < normalizedElText.length) {
			const rawCursor = mapNormalizedIndexToRaw(elText, cursor);
			fragment.appendChild(document.createTextNode(elText.slice(rawCursor)));
		}

		el.textContent = '';
		el.appendChild(fragment);

		flatCursor += normalizedElText.length + 1;
	});
}
