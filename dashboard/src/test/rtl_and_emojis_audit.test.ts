import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('Design Integrity & Quality Bar Audit (Sections 8 & 9)', () => {
  const srcDir = path.resolve(__dirname, '../');

  function getFiles(dir: string, ext: string[]): string[] {
    let results: string[] = [];
    const list = fs.readdirSync(dir);
    list.forEach((file) => {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      if (stat && stat.isDirectory()) {
        if (!file.includes('node_modules') && !file.includes('dist')) {
          results = results.concat(getFiles(fullPath, ext));
        }
      } else {
        if (ext.some((e) => file.endsWith(e))) {
          results.push(fullPath);
        }
      }
    });
    return results;
  }

  it('guarantees ZERO emojis across all source files', () => {
    const files = getFiles(srcDir, ['.ts', '.tsx', '.css']);
    // Official Unicode Extended_Pictographic property matching actual emojis
    const emojiRegex = /\p{Extended_Pictographic}/gu;

    const filesWithEmojis: { file: string; matches: string[] }[] = [];

    files.forEach((file) => {
      // Exclude this test file itself
      if (file.endsWith('rtl_and_emojis_audit.test.ts')) return;
      const content = fs.readFileSync(file, 'utf-8');
      const matches = content.match(emojiRegex);
      if (matches) {
        filesWithEmojis.push({ file: path.relative(srcDir, file), matches });
      }
    });

    expect(filesWithEmojis).toEqual([]);
  });

  it('guarantees ZERO physical direction CSS classes in component code', () => {
    const componentFiles = getFiles(path.join(srcDir, 'components'), ['.tsx']);
    componentFiles.push(path.join(srcDir, 'App.tsx'));

    // Physical direction Tailwind patterns (constructed to keep source grep-clean)
    const prefixes = [
      'm' + 'l-',
      'm' + 'r-',
      'p' + 'l-',
      'p' + 'r-',
      'l' + 'eft-',
      'r' + 'ight-',
      'text-' + 'left',
      'text-' + 'right',
      'border-' + 'l-',
      'border-' + 'r-',
      'rounded-' + 'l-',
      'rounded-' + 'r-',
    ];
    const physicalDirPattern = new RegExp(
      '\\b(' + prefixes.map((p) => p.replace('-', '\\-')).join('|') + ')',
      'g'
    );

    const violations: { file: string; violations: string[] }[] = [];

    componentFiles.forEach((file) => {
      const content = fs.readFileSync(file, 'utf-8');
      const matches = content.match(physicalDirPattern);
      if (matches) {
        violations.push({ file: path.relative(srcDir, file), violations: matches });
      }
    });

    expect(violations).toEqual([]);
  });
});
