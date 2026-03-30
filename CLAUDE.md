# CLAUDE.md

## Commit Rules
- Do NOT add "Co-Authored-By" trailers to commits
- Do NOT add AI attribution (e.g., "Generated with Claude") to code or commit messages

## Testing Rules
- Do NOT use mocks, stubs, fakes, spies, or test doubles of any kind
- No unittest.mock, MagicMock, monkeypatch, jest.mock, vi.mock, jest.fn, sinon, nock, msw
- All tests must hit real databases, real APIs, real services
- If a test can't run against the real thing, it's not a valid test
