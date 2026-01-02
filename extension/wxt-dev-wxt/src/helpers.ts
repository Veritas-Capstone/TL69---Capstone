/* don't look at this, chatGPT wrote it all and I do not know what it does 🔥🔥🔥 */

export function underlineSentences(sentences: { text?: string; claim?: string; valid: boolean }[]) {
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

	/**
	 * 🔥 Deep unwrap inline tags (handles <a><em>text</em></a>)
	 */
	function unwrapInlineTags(root: ParentNode) {
		const nodes = root.querySelectorAll('a, em, strong, cite, span');

		nodes.forEach((node) => {
			// 🚫 Ignore anything inside <nav>
			if (node.closest('nav')) return;

			const parent = node.parentNode;
			if (!parent) return;

			while (node.firstChild) {
				parent.insertBefore(node.firstChild, node);
			}
			parent.removeChild(node);
		});
	}

	// 1️⃣ Select candidate containers FIRST
	const containers = Array.from(
		document.querySelectorAll<HTMLElement>(
			'p, h1, h2, h3, h4, h5, h6, li, span, figcaption, blockquote, footer, aside, cite'
		)
	).filter((el) => !el.closest('script, style'));

	// 2️⃣ UNWRAP INLINE TAGS BEFORE FILTERING
	containers.forEach(unwrapInlineTags);

	// 3️⃣ Now safely find real text elements
	const textElements = containers.filter((el) =>
		Array.from(el.childNodes).some((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim())
	);

	// 4️⃣ Flatten text
	const elementTexts = textElements.map((el) => el.textContent || '');
	const flatText = elementTexts.join('\n');
	const normalizedFlat = normalize(flatText);

	// 5️⃣ Match sentences
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

	// 6️⃣ Underline pass
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
			span.style.textDecoration = 'underline';
			span.style.textDecorationColor = valid ? 'rgba(74, 222, 128, 0.8)' : 'rgba(244, 63, 94, 0.8)';
			span.textContent = elText.slice(rawLocalStart, rawEnd);

			span.addEventListener('mouseenter', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx });
			});
			span.addEventListener('mouseleave', () => {
				browser.runtime.sendMessage({ type: 'UNDERLINE_HOVER', idx: undefined });
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
}
