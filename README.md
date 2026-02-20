
## Context Engineering 따라하기

00_ClaudeCodeCELibrary 아주 기본 템플릿

#### claude code 사용

python 일 경우 가상환경 만들어야 함.
```
python -m venv .venv && .venv\Scripts\activate
```

vs Code에서 좌측 상단에 클로드 아이콘 클릭 or 터미널에서 "claude" 명령


최초 시작 시에만 /init, 명령어 종료된 후 다시 시작 시에는 /primer 
```
/init

or

/primer
```


1. CLAUDE.md 만들어지면 템플릿에서 가져와서 덮어쓰기

2. INITIAL.md 작성 (요청기능 이라고 보면 됨. 원하는 기능 적기) 

3. .claude 폴더 모두 템플릿에서 가져오기 ( command, agent, hooks) : 지속적으로 템플릿 업데이트

4. **examples 폴더** : 아주 중요한 폴더 실제 기능 구현이 제대로 되는 샘플이 들어가면 매우좋음. ui관련해서 프레임 제공하면 좋음.


위에 까지 준비 되었으면 시작하자.

1/ 요구사항 분석 및 개발 사양서 작성 
```
generate-prp INITIAL.md     # 요청되는 기능으로 요구분석서 작성
```
결과물 :  PRPs 폴더에  요구되는 기능과 비스 무리한  "xxx_xxx_xxx_prp.md" 같은 결과파일 생성

2/ 개발 사양서를 바탕으로 실제 구현 
```
execute-prp xxx_xxx_xxx_prp.md
```

1, 또는 2번 을 선택후 엔터 쳐서 끝까지 가면 됨.

끝,

테스트 및 유지보수 (얼마나 더 해야 될지 모름, 첨부터 잘 준비하면? 특히 samples 폴더의 코드가 제일 중요)
