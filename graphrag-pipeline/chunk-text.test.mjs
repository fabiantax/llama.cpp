/**
 * Property-based tests for chunkText() using fast-check.
 *
 * Run: npm install --save-dev fast-check && node graphrag-pipeline/chunk-text.test.mjs
 */

import fc from 'fast-check';

// ── Extract chunkText from extract.mjs (copy to keep test isolated) ──────

const CHUNK_SIZE = 600;
const CHUNK_OVERLAP = 100;

function chunkText(text, chunkChars = CHUNK_SIZE * 4, overlapChars = CHUNK_OVERLAP * 4) {
  const chunks = [];
  let start = 0;
  const maxIterations = text.length + 1;

  for (let iter = 0; iter < maxIterations && start < text.length; iter++) {
    let end = Math.min(start + chunkChars, text.length);
    if (end < text.length) {
      const slice = text.slice(start, end);
      const lastPara = slice.lastIndexOf('\n\n');
      const lastSentence = slice.lastIndexOf('. ');
      if (lastPara > chunkChars * 0.5) end = start + lastPara + 2;
      else if (lastSentence > chunkChars * 0.5) end = start + lastSentence + 2;
    }
    chunks.push(text.slice(start, end));
    if (end >= text.length) break;
    const newStart = end > overlapChars ? end - overlapChars : end;
    start = Math.max(newStart, start + 1);
  }
  return chunks;
}

// ── Test helpers ─────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.error(`  FAIL  ${name}`);
    console.error(`        ${e.message}`);
    if (e.counterexample) {
      console.error(`        Counterexample: ${JSON.stringify(e.counterexample)}`);
    }
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed');
}

function assertEqual(a, b, msg) {
  if (JSON.stringify(a) !== JSON.stringify(b))
    throw new Error(msg || `Expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
}

// ── Deterministic edge-case tests ────────────────────────────────────────

console.log('\n=== Deterministic tests ===');

test('empty string returns empty array', () => {
  assertEqual(chunkText(''), []);
});

test('single char', () => {
  assertEqual(chunkText('x'), ['x']);
});

test('text shorter than overlap (original infinite loop trigger)', () => {
  const chunks = chunkText('short', 2400, 400);
  assertEqual(chunks, ['short']);
});

test('text exactly chunk size', () => {
  const text = 'a'.repeat(2400);
  const chunks = chunkText(text, 2400, 400);
  assertEqual(chunks, [text]);
});

test('overlap larger than chunk (pathological)', () => {
  const text = 'a'.repeat(100);
  const chunks = chunkText(text, 10, 50);
  assert(chunks.length > 0, 'Must produce chunks');
  assert(chunks.length <= text.length, `Too many chunks: ${chunks.length}`);
  assert(chunks[0][0] === 'a', 'First chunk starts correctly');
  assert(chunks[chunks.length - 1].endsWith('a'), 'Last chunk ends correctly');
});

test('paragraph boundary splitting', () => {
  const text = 'a'.repeat(1500) + '.\n\n' + 'b'.repeat(1500);
  const chunks = chunkText(text, 2400, 400);
  assert(chunks.length >= 2, 'Should split at paragraph');
  assert(chunks[0].endsWith('\n\n'), 'First chunk should end at paragraph boundary');
});

test('sentence boundary splitting', () => {
  const text = 'a'.repeat(1500) + '. ' + 'b'.repeat(1500);
  const chunks = chunkText(text, 2400, 400);
  assert(chunks.length >= 2, 'Should split at sentence');
  assert(chunks[0].endsWith('. '), 'First chunk should end at sentence boundary');
});

test('all newlines (dense boundaries)', () => {
  const text = '\n\n'.repeat(500);
  const chunks = chunkText(text, 100, 50);
  assert(chunks.length > 0, 'Must produce chunks');
  assert(chunks.length < text.length, 'Should not explode into too many chunks');
});

test('unicode text', () => {
  const text = '日本語のテスト。'.repeat(300);
  const chunks = chunkText(text, 100, 20);
  assert(chunks.length > 0, 'Must produce chunks');
  assert(chunks[0].startsWith('日'), 'First chunk starts correctly');
  assert(chunks[chunks.length - 1].endsWith('。'), 'Last chunk ends correctly');
});

test('full coverage: chunks span from start to end without gaps', () => {
  // Use non-repeating text so indexOf is unambiguous
  const text = Array.from({ length: 500 }, (_, i) => `w${i} `).join('');
  const chunks = chunkText(text, 100, 20);
  // Verify: first chunk starts at 0, last chunk ends at text.length
  assert(text.startsWith(chunks[0]), 'First chunk must be a prefix');
  assert(text.endsWith(chunks[chunks.length - 1]), 'Last chunk must be a suffix');
  // Verify sequential overlap: each chunk's start must be within the previous chunk
  let prevEnd = chunks[0].length;
  for (let i = 1; i < chunks.length; i++) {
    const pos = text.indexOf(chunks[i]);
    assert(pos >= 0, `Chunk ${i} not found in text`);
    assert(pos < prevEnd, `Gap between chunk ${i - 1} (ends ${prevEnd}) and chunk ${i} (starts ${pos})`);
    prevEnd = pos + chunks[i].length;
  }
  assert(prevEnd === text.length, `Chunks don't reach end: ${prevEnd} vs ${text.length}`);
});

// ── Property-based tests (fast-check) ────────────────────────────────────

console.log('\n=== Property-based tests (fast-check) ===');

test('terminates and returns non-empty for non-empty input', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 5000 }),
    fc.integer({ min: 1, max: 5000 }),
    fc.integer({ min: 0, max: 5000 }),
    (text, chunkChars, overlapChars) => {
      const chunks = chunkText(text, chunkChars, overlapChars);
      return chunks.length > 0;
    }
  ), { numRuns: 1000 });
});

test('first chunk is a prefix of the text', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 5000 }),
    fc.integer({ min: 1, max: 5000 }),
    fc.integer({ min: 0, max: 5000 }),
    (text, chunkChars, overlapChars) => {
      const chunks = chunkText(text, chunkChars, overlapChars);
      return text.startsWith(chunks[0]);
    }
  ), { numRuns: 1000 });
});

test('last chunk is a suffix of the text', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 5000 }),
    fc.integer({ min: 1, max: 5000 }),
    fc.integer({ min: 0, max: 5000 }),
    (text, chunkChars, overlapChars) => {
      const chunks = chunkText(text, chunkChars, overlapChars);
      return text.endsWith(chunks[chunks.length - 1]);
    }
  ), { numRuns: 1000 });
});

test('chunk count is bounded by text length', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 5000 }),
    fc.integer({ min: 1, max: 5000 }),
    fc.integer({ min: 0, max: 5000 }),
    (text, chunkChars, overlapChars) => {
      const chunks = chunkText(text, chunkChars, overlapChars);
      return chunks.length <= text.length;
    }
  ), { numRuns: 1000 });
});

test('every chunk is a substring of the original text', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 3000 }),
    fc.integer({ min: 10, max: 3000 }),
    fc.integer({ min: 0, max: 1000 }),
    (text, chunkChars, overlapChars) => {
      const chunks = chunkText(text, chunkChars, overlapChars);
      return chunks.every(c => text.includes(c));
    }
  ), { numRuns: 1000 });
});

test('no chunk is empty', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 5000 }),
    fc.integer({ min: 1, max: 5000 }),
    fc.integer({ min: 0, max: 5000 }),
    (text, chunkChars, overlapChars) => {
      const chunks = chunkText(text, chunkChars, overlapChars);
      return chunks.every(c => c.length > 0);
    }
  ), { numRuns: 1000 });
});

test('extreme overlap: overlap = 10x chunk', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 1000 }),
    fc.integer({ min: 1, max: 100 }),
    (text, chunkChars) => {
      const overlapChars = chunkChars * 10;
      const chunks = chunkText(text, chunkChars, overlapChars);
      return chunks.length > 0 && chunks.length <= text.length;
    }
  ), { numRuns: 500 });
});

test('chunk_chars = 1 (minimum possible chunk size)', () => {
  fc.assert(fc.property(
    fc.string({ minLength: 1, maxLength: 500 }),
    fc.integer({ min: 0, max: 500 }),
    (text, overlapChars) => {
      const chunks = chunkText(text, 1, overlapChars);
      return chunks.length > 0 && chunks.length <= text.length;
    }
  ), { numRuns: 500 });
});

// ── Summary ──────────────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
