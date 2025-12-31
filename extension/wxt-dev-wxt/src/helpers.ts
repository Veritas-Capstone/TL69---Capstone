export function underlineSentences(sentences: { text: string; valid: boolean; claim: string }[]) {
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
			const start = normalizedFlat.indexOf(normalize(s.text ?? s.claim));
			if (start === -1) return null;
			return {
				idx,
				valid: s.valid,
				start,
				end: start + normalize(s.text ?? s.claim).length,
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
			span.style.paddingTop = '4px';
			span.style.paddingBottom = '4px';
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
