# Contributing Guide

이 프로젝트에 기여해주셔서 감사합니다! 아래 규칙을 따라 협업을 진행해주세요.

## 목차
- [브랜치 전략](#브랜치-전략)
- [커밋 메시지 컨벤션](#커밋-메시지-컨벤션)
- [Pull Request 규칙](#pull-request-규칙)
- [Issue 작성 규칙](#issue-작성-규칙)
- [코드 리뷰 규칙](#코드-리뷰-규칙)
- [코딩 컨벤션](#코딩-컨벤션)

---

## 브랜치 전략

### 브랜치 종류

| 브랜치 | 설명 |
|---|---|
| `main` | 배포 가능한 안정 버전 |
| `develop` | 다음 배포를 위한 개발 브랜치 |
| `feature/*` | 기능 개발 브랜치 |
| `fix/*` | 버그 수정 브랜치 |
| `hotfix/*` | 운영 환경 긴급 수정 브랜치 |
| `refactor/*` | 리팩토링 브랜치 |
| `docs/*` | 문서 작업 브랜치 |
| `chore/*` | 빌드, 설정 등 기타 작업 브랜치 |

### 브랜치 네이밍 규칙

```
<type>/<본인 이름>-<간단한-설명>
```

**예시**
```
feature/gildong-login-api
fix/gildong-null-pointer-exception
docs/gildong-update-readme
```

---

## 커밋 메시지 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/) 규칙을 따릅니다.

### 형식

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **subject**: 50자 이내, 마침표 없이, 명령형으로 작성 (예: "추가하다" O, "추가함" X)
- **body**: 변경 이유와 이전과 달라진 점을 설명 (선택 사항, 72자 줄바꿈 권장)
- **footer**: 이슈 트래커 참조 (`Closes #123`, `Related to #45`)

### Type 종류

| Type | 설명 |
|---|---|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 포맷팅, 세미콜론 누락 등 (로직 변경 없음) |
| `refactor` | 코드 리팩토링 (기능 변화 없음) |
| `test` | 테스트 코드 추가/수정 |
| `chore` | 빌드 업무, 패키지 매니저 설정 등 |
| `perf` | 성능 개선 |
| `ci` | CI 설정 파일/스크립트 변경 |
| `revert` | 이전 커밋 되돌리기 |

### 커밋 예시

```
feat(auth): 소셜 로그인 기능 추가

카카오/네이버 OAuth2 로그인 연동
기존 이메일 로그인과 동일한 세션 정책 적용

Closes #12
```

```
fix(cart): 수량 0일 때 장바구니 삭제 안되는 버그 수정
```

### 커밋 시 주의사항
- 하나의 커밋에는 하나의 논리적 변경사항만 담습니다.
- 커밋은 작고 자주 나눕니다 (기능 단위가 아닌 작업 단위).
- WIP(작업 중) 커밋은 PR 전에 `rebase -i`로 정리합니다.

---

## Pull Request 규칙

### PR 제목 형식

커밋 컨벤션과 동일하게 작성합니다.

```
<type>(<scope>): <설명>
```

예: `feat(auth): 소셜 로그인 기능 추가`

### PR 크기
- 한 PR은 가능한 500줄 이하로 작게 유지합니다.
- 리뷰가 어려울 정도로 크다면 여러 PR로 분리합니다.

### PR 절차
1. `develop`(또는 `main`)에서 작업 브랜치 생성
2. 작업 완료 후 원격 브랜치에 push
3. PR 템플릿(`.github/PULL_REQUEST_TEMPLATE.md`)에 맞춰 작성
4. 최소 1명 이상의 리뷰어 승인(Approve) 후 머지
5. 머지 방식은 **Squash and Merge**를 기본으로 함
6. 머지 후 작업 브랜치는 삭제

---

## Issue 작성 규칙

### Issue 제목 형식

```
[TYPE] 간단한 설명
```

예:
```
[BUG] 로그인 시 500 에러 발생
[FEATURE] 다크모드 지원 요청
```

### 라벨(Label) 규칙

| 라벨 | 설명 |
|---|---|
| `bug` | 버그 리포트 |
| `feature` | 신규 기능 요청 |
| `enhancement` | 기존 기능 개선 |
| `documentation` | 문서 관련 |
| `question` | 질문 |
| `wontfix` | 처리하지 않을 이슈 |
| `duplicate` | 중복 이슈 |
| `priority: high/medium/low` | 우선순위 |

Issue 템플릿은 `.github/ISSUE_TEMPLATE/` 폴더의 `bug_report.md`, `feature_request.md`를 참고해주세요.

---

## 코드 리뷰 규칙

- 리뷰는 24시간 이내 응답을 원칙으로 합니다.
- 리뷰 코멘트는 구체적이고 건설적으로 작성합니다.
- Approve 전 반드시 로컬에서 동작을 확인하거나 CI 통과를 확인합니다.
- 사소한 스타일 지적은 `nit:` 접두사를 붙입니다 (머지를 막지 않는 의견임을 표시).

```
nit: 변수명을 좀 더 명확하게 바꾸면 좋을 것 같아요.
```

---

## 코딩 컨벤션

- 프로젝트에서 사용 중인 Linter/Formatter(ESLint, Prettier 등) 설정을 그대로 따릅니다.
- 커밋 전 반드시 `lint`, `format`, `test`를 로컬에서 실행합니다.
- 매직 넘버, 하드코딩된 값은 상수로 분리합니다.
- 함수/변수명은 의미가 명확하게 드러나도록 작성합니다.

---

## 질문이 있다면

Issue 또는 팀 채널을 통해 언제든 편하게 문의해주세요.
