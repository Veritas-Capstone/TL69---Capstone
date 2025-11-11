import { defineConfig } from 'wxt';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

// See https://wxt.dev/api/config.html
export default defineConfig({
	modules: ['@wxt-dev/module-react', '@wxt-dev/auto-icons'],
	vite: () => ({
		plugins: [tailwindcss()],
		resolve: {
			alias: {
				'@': path.resolve(__dirname, './'), // or "./src" if using src directory
			},
		},
	}),
	autoIcons: {
		developmentIndicator: false,
		sizes: [16, 32, 48, 96, 128],
	},
	manifest: {
		permissions: ['contextMenus', 'storage', 'scripting', 'activeTab', 'sidePanel'],
		name: 'Veritas',
		icons: {
			16: 'icons/16.png',
			32: 'icons/32.png',
			48: 'icons/48.png',
			96: 'icons/96.png',
			128: 'icons/128.png',
		},
		side_panel: {
			default_path: 'entrypoints/sidepanel.html',
		},
		content_scripts: [
			{
				matches: ['<all_urls>'],
				js: ['entrypoints/content.ts'],
			},
		],
	},
	srcDir: 'src',
});
