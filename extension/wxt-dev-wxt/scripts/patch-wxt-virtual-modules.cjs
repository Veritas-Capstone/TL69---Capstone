const fs = require('fs');
const path = require('path');

const target = path.join(
	__dirname,
	'..',
	'node_modules',
	'wxt',
	'dist',
	'core',
	'builders',
	'vite',
	'plugins',
	'resolveVirtualModules.mjs',
);

if (!fs.existsSync(target)) {
	console.warn(`[patch-wxt] Skipping: ${target} not found`);
	process.exit(0);
}

const source = fs.readFileSync(target, 'utf8');
const before = "return template.replace(`virtual:user-${name}`, inputPath);";
const after = "return template.replace(`'virtual:user-${name}'`, JSON.stringify(inputPath));";

if (source.includes(after)) {
	console.log('[patch-wxt] WXT virtual module path patch already applied');
	process.exit(0);
}

if (!source.includes(before)) {
	console.warn('[patch-wxt] Expected WXT source pattern not found; no changes made');
	process.exit(0);
}

fs.writeFileSync(target, source.replace(before, after), 'utf8');
console.log('[patch-wxt] Patched WXT virtual module path escaping');
