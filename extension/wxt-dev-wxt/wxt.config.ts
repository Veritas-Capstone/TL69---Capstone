import { defineConfig } from 'wxt';

// See https://wxt.dev/api/config.html
export default defineConfig({
	modules: ['@wxt-dev/module-react', '@wxt-dev/auto-icons'],
	autoIcons: {
		developmentIndicator: false,
		sizes: [16, 32, 48, 96, 128],
	},
	manifest: {
		permissions: ['contextMenus', 'storage'],
		name: 'Veritas',
		icons: {
			16: 'icons/16.png',
			32: 'icons/32.png',
			48: 'icons/48.png',
			96: 'icons/96.png',
			128: 'icons/128.png',
		},
	},
	srcDir: 'src',
});
