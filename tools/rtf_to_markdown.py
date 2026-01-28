#!/usr/bin/env python3
"""
RTF to Markdown Converter

Algorithmically converts RTF files to plaintext, then structures as Markdown.
"""

import re
import os
import sys
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ConversionResult:
    """Result of RTF to Markdown conversion."""
    source_path: str
    markdown_path: str
    plaintext: str
    markdown: str
    success: bool
    error: Optional[str] = None


class RTFToMarkdown:
    """Algorithmic RTF to Markdown converter."""

    # RTF control words for groups to skip entirely
    SKIP_GROUPS = {
        'fonttbl', 'colortbl', 'stylesheet', 'listtable', 'listoverridetable',
        'info', 'header', 'footer', 'pict', 'object', 'expandedcolortbl',
        'generator', 'panose', 'bkmkstart', 'bkmkend', 'field', 'fldinst',
        'levelnumbers', 'leveltext', 'listname',
    }

    # Special characters
    SPECIAL_CHARS = {
        'bullet': '•',
        'endash': '–',
        'emdash': '—',
        'lquote': ''',
        'rquote': ''',
        'ldblquote': '"',
        'rdblquote': '"',
        'tab': '\t',
    }

    def decode_unicode(self, text: str) -> str:
        """Decode RTF unicode escapes."""
        def replace_unicode(m):
            try:
                code = int(m.group(1))
                if code < 0:
                    code += 65536
                return chr(code)
            except:
                return ''
        return re.sub(r'\\u(-?\d+)[^\\a-zA-Z]?', replace_unicode, text)

    def decode_hex(self, text: str) -> str:
        """Decode RTF hex escapes."""
        def replace_hex(m):
            try:
                return chr(int(m.group(1), 16))
            except:
                return ''
        return re.sub(r"\\'([0-9a-fA-F]{2})", replace_hex, text)

    def strip_rtf(self, rtf: str) -> str:
        """Extract plaintext from RTF content."""
        text = rtf

        # Remove RTF header
        text = re.sub(r'^\{\\rtf1[^\n]*', '', text)

        # Remove entire list table blocks - these are complex nested structures
        # Match from {\*\listtable to the matching closing brace
        depth = 0
        result = []
        i = 0
        in_listtable = False
        skip_depth = 0

        while i < len(text):
            # Check for list table start
            if text[i:i+13] == '{\\*\\listtable':
                in_listtable = True
                skip_depth = 1
                i += 13
                continue
            elif text[i:i+21] == '{\\*\\listoverridetable':
                in_listtable = True
                skip_depth = 1
                i += 21
                continue

            if in_listtable:
                if text[i] == '{':
                    skip_depth += 1
                elif text[i] == '}':
                    skip_depth -= 1
                    if skip_depth == 0:
                        in_listtable = False
                i += 1
                continue

            result.append(text[i])
            i += 1

        text = ''.join(result)

        # Remove patterns like disc-360, decimal-360, circle-360 (list markers that leak through)
        text = re.sub(r'\b(?:disc|decimal|circle|square)-?\d+\b', '', text)
        # Remove remaining list-related control words (include backslash to avoid orphans)
        text = re.sub(r'\\(?:ls\d+|levelnfc\d+|leveltext)\s?', '', text)
        text = re.sub(r'\\ilvl\d+\s?', '', text)

        # Remove listtext markers and their content (these are list item numbers)
        text = re.sub(r'\{\\listtext[^}]*\}', '', text)

        # Remove groups we want to skip (font tables, color tables, etc.)
        for group in self.SKIP_GROUPS:
            # Simple pattern - matches {\groupname ...}
            text = re.sub(r'\{[^{}]*\\' + group + r'[^{}]*\}', '', text, flags=re.DOTALL)

        # Remove deeply nested skip groups
        for _ in range(5):  # Multiple passes for nested structures
            for group in self.SKIP_GROUPS:
                text = re.sub(r'\{[^{}]*\\' + group + r'(?:[^{}]|\{[^{}]*\})*\}', '', text, flags=re.DOTALL)

        # Decode unicode and hex escapes
        text = self.decode_unicode(text)
        text = self.decode_hex(text)

        # Handle special symbols
        for sym, char in self.SPECIAL_CHARS.items():
            text = re.sub(r'\\' + sym + r'\b\s?', char, text)

        # Replace paragraph markers with newlines
        text = re.sub(r'\\par\b\s?', '\n', text)
        text = re.sub(r'\\line\b\s?', '\n', text)
        text = re.sub(r'\\\\\s?', '\n', text)

        # Handle table cells - convert to tab separation
        text = re.sub(r'\\cell\b\s?', '\t', text)
        text = re.sub(r'\\row\b\s?', '\n', text)

        # Remove remaining control words (including negative numeric args like \fi-720)
        text = re.sub(r'\\[a-zA-Z]+-?\d*\s?', '', text)

        # Remove escaped characters
        text = re.sub(r'\\[{}~_\-\*]', '', text)
        text = text.replace('\\', '')

        # Remove braces
        text = text.replace('{', '').replace('}', '')

        # Clean whitespace
        text = re.sub(r'[ ]+', ' ', text)
        text = re.sub(r'\t+', '\t', text)

        # Remove common RTF artifacts that may leak through
        text = re.sub(r'^-?\s*\d{2,4}\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers like "108"
        text = re.sub(r'^\s*-\s*\d{2,4}\s*$', '', text, flags=re.MULTILINE)  # "- 108" patterns
        text = re.sub(r'^\s*\d+[gaph]\d*\s*$', '', text, flags=re.MULTILINE)  # gaph patterns

        # Remove binary control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Remove lines that are just semicolons, numbers with semicolons, or RTF artifacts
        text = re.sub(r'^\s*;+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*;-?\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*-?\d+;?\s*$', '', text, flags=re.MULTILINE)

        # Remove remaining level/list artifacts
        text = re.sub(r'\blevel\w+\d*\b', '', text)
        text = re.sub(r'\blist\w+\d*\b', '', text)
        text = re.sub(r'\bilvl\d+\b', '', text)

        # Clean up orphaned punctuation and fragments
        text = re.sub(r'^\s*[;◦•]\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\(\s*\)\s*$', '', text, flags=re.MULTILINE)

        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Join fragmented lines - lines that are continuations of previous content
        # Only join consecutive non-empty lines (respect blank line separators)
        result_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                result_lines.append('')
                continue

            # Check if this line should be joined with the previous non-empty line
            if result_lines and result_lines[-1]:
                prev = result_lines[-1]
                prev_ends_sentence = prev.endswith(('.', '!', '?', '"', ')'))
                prev_ends_partial = prev.endswith((',', ';', ':', '(', '–', '—'))

                is_continuation = False

                # Starts with lowercase - likely continuation
                if line[0].islower():
                    is_continuation = True
                # Starts with continuation punctuation
                elif line[0] in '–—,;:)].':
                    is_continuation = True
                # Previous ends with partial punctuation
                elif prev_ends_partial:
                    is_continuation = True
                # Previous doesn't end sentence and current is short
                elif not prev_ends_sentence and len(line) < 30 and not line[0].isdigit():
                    is_continuation = True

                if is_continuation:
                    if line[0] in '.,;:)!?':
                        result_lines[-1] = prev + line
                    else:
                        result_lines[-1] = prev + ' ' + line
                    continue

            result_lines.append(line)

        text = '\n'.join(result_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _is_artifact_line(self, line: str) -> bool:
        """Check if a line is an RTF artifact that should be skipped."""
        line = line.strip()
        # Skip empty or whitespace-only lines
        if not line:
            return True
        # Skip lines that are just punctuation, numbers, or RTF fragments
        if re.match(r'^[\s;,.\-•◦\d]*$', line):
            return True
        if re.match(r'^(ilvl|level|list)\w*\d*$', line, re.IGNORECASE):
            return True
        # Skip single letters or letter-number combos (formatting artifacts like b0, i0)
        if re.match(r'^[a-z]\d*$', line):
            return True
        # Skip very short non-word content
        if len(line) < 3 and not re.match(r'^[A-Za-z]{2,}$', line):
            return True
        # Skip arrow-only lines
        if line in ['→', '—', '–', '•', '◦', ')']:
            return True
        return False

    def to_markdown(self, plaintext: str, filename: str = "") -> str:
        """Convert plaintext to Markdown with structure detection."""
        lines = plaintext.split('\n')
        md = []

        # Title from filename
        title = Path(filename).stem if filename else ""
        title = re.sub(r'^\d+[._]?\s*', '', title)  # Remove number prefix
        title = title.replace('_', ' ').replace('-', ' ').strip()
        if title:
            md.append(f"# {title.title()}")
            md.append("")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                if md and md[-1] != "":
                    md.append("")
                i += 1
                continue

            # Skip artifact lines
            if self._is_artifact_line(line):
                i += 1
                continue

            # Detect tables (tab-separated content)
            if '\t' in line:
                table_lines = []
                while i < len(lines) and ('\t' in lines[i] or not lines[i].strip()):
                    if lines[i].strip() and not self._is_artifact_line(lines[i]):
                        table_lines.append(lines[i].strip())
                    i += 1
                if table_lines:
                    md.extend(self._make_table(table_lines))
                    md.append("")
                continue

            # Detect bullet lists
            bullet_match = re.match(r'^[•\-\*\u2022\u25E6]\s*(.+)', line)
            if bullet_match:
                content = bullet_match.group(1)
                if len(content) > 2:  # Skip if content is too short
                    md.append(f"- {content}")
                i += 1
                continue

            # Detect numbered lists
            num_match = re.match(r'^(\d+)[.)]\s*(.+)', line)
            if num_match:
                content = num_match.group(2)
                if len(content) > 2:  # Skip if content is too short
                    md.append(f"{num_match.group(1)}. {content}")
                    i += 1
                    continue

            # Detect headers (more conservative heuristics)
            if len(line) < 80 and i + 1 < len(lines) and len(line) > 5:
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                # Only treat as header if it looks like a real title
                is_title_case = line[0].isupper() and not line.isupper()
                has_content_words = len([w for w in line.split() if len(w) > 2]) >= 2

                # Section header patterns - numbered sections
                if re.match(r'^\d+\.\s+[A-Z]', line) and len(line) < 60 and has_content_words:
                    md.append(f"## {line}")
                    md.append("")
                    i += 1
                    continue
                # Questions as headers
                elif line.endswith('?') and len(line) < 60 and len(line) > 10:
                    md.append(f"## {line}")
                    md.append("")
                    i += 1
                    continue
                # Short line followed by much longer line = likely header
                # But only if the short line looks like a title
                elif (len(line) < 40 and len(line) > 8 and next_line
                      and len(next_line) > len(line) * 2
                      and is_title_case and has_content_words):
                    md.append(f"## {line}")
                    md.append("")
                    i += 1
                    continue

            # Regular paragraph - but skip very short lines that aren't meaningful
            if len(line) > 3 or line.isalpha():
                md.append(line)
            i += 1

        result = '\n'.join(md)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    def _make_table(self, lines: List[str]) -> List[str]:
        """Convert tab-separated lines to markdown table."""
        rows = []
        for line in lines:
            cols = [c.strip() for c in line.split('\t') if c.strip()]
            if cols:
                rows.append(cols)

        if not rows:
            return lines

        # Normalize column count
        max_cols = max(len(r) for r in rows)
        rows = [r + [''] * (max_cols - len(r)) for r in rows]

        md = []
        md.append('| ' + ' | '.join(rows[0]) + ' |')
        md.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
        for row in rows[1:]:
            md.append('| ' + ' | '.join(row) + ' |')
        return md

    def convert_file(self, rtf_path: str, output_dir: Optional[str] = None) -> ConversionResult:
        """Convert a single RTF file to Markdown."""
        rtf_path = Path(rtf_path)

        try:
            # Read with error handling for encoding issues
            with open(rtf_path, 'rb') as f:
                raw = f.read()

            # Try UTF-8, fall back to latin-1
            try:
                rtf_content = raw.decode('utf-8')
            except UnicodeDecodeError:
                rtf_content = raw.decode('latin-1')

            # Convert
            plaintext = self.strip_rtf(rtf_content)
            markdown = self.to_markdown(plaintext, rtf_path.name)

            # Output path
            if output_dir:
                output_path = Path(output_dir) / (rtf_path.stem + '.md')
            else:
                output_path = rtf_path.with_suffix('.md')

            # Write (handle encoding issues)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            clean_markdown = markdown.encode('utf-8', errors='replace').decode('utf-8')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(clean_markdown)

            return ConversionResult(
                source_path=str(rtf_path),
                markdown_path=str(output_path),
                plaintext=plaintext,
                markdown=clean_markdown,
                success=True
            )

        except Exception as e:
            return ConversionResult(
                source_path=str(rtf_path),
                markdown_path="",
                plaintext="",
                markdown="",
                success=False,
                error=str(e)
            )

    def convert_directory(self, dir_path: str, output_dir: Optional[str] = None,
                         recursive: bool = True) -> List[ConversionResult]:
        """Convert all RTF files in a directory."""
        dir_path = Path(dir_path)
        results = []

        pattern = '**/*.rtf' if recursive else '*.rtf'
        for rtf_file in sorted(dir_path.glob(pattern)):
            # Skip archives
            if 'archive' in str(rtf_file).lower():
                continue

            result = self.convert_file(rtf_file, output_dir)
            results.append(result)
            status = "✓" if result.success else "✗"
            print(f"{status} {rtf_file.name}")

        return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python rtf_to_markdown.py <path> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    converter = RTFToMarkdown()

    if input_path.is_file():
        result = converter.convert_file(input_path, output_dir)
        if result.success:
            print(f"Converted: {result.markdown_path}")
        else:
            print(f"Error: {result.error}")
    elif input_path.is_dir():
        results = converter.convert_directory(input_path, output_dir)
        success = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        print(f"\nConverted {success} files, {failed} errors")
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
