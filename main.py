# -*- coding: utf-8 -*-
"""
Mini NPU Simulator
- MAC(Multiply-Accumulate) 연산을 반복문으로 직접 구현 (외부 라이브러리 미사용)
- 모드 1: 사용자 입력(3x3) 기반 A/B 필터 판정
- 모드 2: data.json 기반 일괄 판정 + 스키마 검증 + 라벨 정규화 + PASS/FAIL 요약
- 성능 분석: 크기별(N x N) MAC 연산 시간(ms) 측정
"""

import json
import time
from typing import List, Tuple, Dict, Any, Optional

EPSILON = 1e-9  # 점수 비교 허용 오차,0.000000001


# ============================================================
# 데이터 구조
# ============================================================
class Grid:
    """N x N 2차원 데이터(패턴/필터)를 저장하는 클래스.

    - 특정 위치(row, col)의 값을 get/set으로 읽고 쓸 수 있다.
    - 내부적으로는 표준 파이썬 2차원 리스트(list of list)를 사용한다.
    """

    def __init__(self, size: int, data: Optional[List[List[float]]] = None):#data값이 안넘어오면 none으로 초기화,함수에서 사용하면 대입이 아니라 기본지정이 된다,나머지는 타입 힌트
        self.size = size 
        if data is not None:
            self.data = data
        else: #data = None이면 0.0으로 초기화
            self.data = [[0.0 for _ in range(size)] for _ in range(size)] # [[0.0]*size]*size 같은 방식으로 만들면 안에 있는 리스트들이 전부 같은 객체를 참조해서 하나만 바꿔도 전체 행이 다 바뀌는 버그생김

    def get(self, row: int, col: int) -> float: #row행 col열의 값 참고,return타입이 FLOAT임
        return self.data[row][col]

    def set(self, row: int, col: int, value: float) -> None: #row행 col열의 값 대입, return값 없음
        self.data[row][col] = value

    @classmethod  #클래스 매소드,2차원 리스트를 받아서 grid객체로 만들어 반환하는 클래스매소드임-> 왜 클래스 매소드를 썻냐면 json에서 받아온 데이터를 기반으로 grid클래스 객체가 없는 상태에서 grid클래스 객체를 만들어야 하기떄문
    def from_2d_list(cls, values: List[List[float]]) -> "Grid":#cls는 클래스 grid를 의미하고 grid를 써도 무방하나 나중에 grid를 다른 클래스에 상속시킬떄 cls를 쓰지않으면 grid로 고정되어 상속받은 클래스에서 오류 이르킬수도있음
        size = len(values) #size계산
        return cls(size, [[float(v) for v in row] for row in values])#데이터 값을 float 2차원리스트로 변환하여 cls(grid)객체로 반환


# ============================================================
# MAC 연산 (외부 라이브러리 금지, 반복문 직접 구현)
# ============================================================
def mac_operation(pattern: Grid, filt: Grid) -> float:#mac기능 수행
    """MAC(Multiply-Accumulate) 연산.

    입력 패턴과 필터를 같은 위치끼리 곱한 뒤 그 값을 모두 누적해서 더한다.
    score = sum( pattern[i][j] * filter[i][j] )  for all i, j
    """
    if pattern.size != filt.size: #필터랑 사이즈 안맞으면 크기불일치 오류 발생시키기
        raise ValueError(f"크기 불일치: pattern={pattern.size}, filter={filt.size}")

    total = 0.0 #점수
    n = pattern.size 
    for i in range(n):
        for j in range(n):
            total += pattern.get(i, j) * filt.get(i, j) #점수계산 -> 같은 위치(i, j)의 패턴 값과 필터 값을 곱해서 total에 저장
    return total #점수 반환


# ============================================================
# 패턴 생성기 (성능 측정 및 보너스 과제용)
# ============================================================
def generate_cross_pattern(n: int) -> Grid:
    """N x N 크기의 십자가(+,Cross) 패턴 생성"""
    g = Grid(n) #data = none 임으로 0으로 채워져있는 grid생성
    mid = n // 2 #중간파악
    for i in range(n):
        for j in range(n):
            g.set(i, j, 1.0 if (i == mid or j == mid) else 0.0)# 행과 열이 mid와 같으면 1 아니면 0대입, 짝수면 우측 하단? 쪽으로 치우쳐서 만들어짐
    return g #g객체 반환


def generate_x_pattern(n: int) -> Grid: 
    """N x N 크기의 X 패턴 생성"""
    g = Grid(n) #init에서 data =none 임으로 0으로 채워져있는 grid생성
    for i in range(n):
        for j in range(n):
            g.set(i, j, 1.0 if (i == j or i + j == n - 1) else 0.0)#왼쪽에서 오른쪽으로 내려가는 대각선-> i = j이면 생성, 오른쪽에서 왼쪽으로 내려가는 대각선-> i+j =n-1이면 생성 
    return g


# ============================================================
# 라벨 정규화
# ============================================================
def normalize_label(raw: Any) -> str: #raw타입은 미정, return타입 str
    """표준 라벨(Cross, X)로 정규화한다.

    - expected: '+' -> Cross, 'x' -> X
    - filter 키: 'cross' -> Cross, 'x' -> X
    알 수 없는 값이 들어오면 원본 문자열을 그대로 반환한다(호출부에서 오류 처리).
    """
    key = str(raw).strip().lower() #raw 문자열로 바꾸고 공백재거하고 소문자로 전환
    if key in ("+", "cross"): #+,cross -> Cross로 통일
        return "Cross"
    if key == "x": #x -> X로 통일
        return "X"
    return str(raw) #둘다 아니면 입력값 그대로 return


# ============================================================
# 성능 측정,출력
# ============================================================
#시간 측정
def measure_mac_time_ms(pattern: Grid, filt: Grid, repeats: int = 10) -> float:#pattern,filt->Grid 객체,repaets->int 타입으로 기본값 10
    """MAC 연산 시간을 repeats회 반복 측정하여 평균 시간(ms)을 반환한다.

    I/O(입력/출력/파일 읽기)는 제외하고, MAC 연산 함수 호출 구간만 측정한다.
    """
    start = time.perf_counter() #시작 시점 기록,time라이브러리의 정밀 측정 perf_counter()사용
    for _ in range(repeats): #repeats회 반복
        mac_operation(pattern, filt) #mac기능 수행,시간만 측정함으로 return값 불필요
    end = time.perf_counter()#종료시점 기록
    total_ms = (end - start) * 1000.0 #s에서 ms로 변환
    return total_ms / repeats #repeats로 나눠서 평균 소요시간 return

#Grid 크기별로 측정한 MAC 연산 평균 시간과 연산 횟수 표로 출력
def print_perf_table(rows: List[Tuple[int, float]]) -> None: #rows -> (정수,실수)꼴의 튜플로 이루어진 리스트, return값 없음
    print(f"{'크기':<10}{'평균 시간(ms)':<18}{'연산 횟수(N^2)':<14}") # < : 왼쪽정렬, 10: 전체너비 10 나머지 동일
    print("-" * 40)
    for size, avg_ms in rows: # rows튜플의 size와 avg_ms 출력(언패킹,튜플을 변수에 나눠담는거)
        label = f"{size}×{size}" #n*n 꼴 출력
        print(f"{label:<10}{avg_ms:<18.3f}{size * size:<14}")#평균시간 소수점 3자리까지 연산횟수는 n^2 출력


# ============================================================
# 모드 1: 사용자 입력 (3x3)
# ============================================================
def input_grid(prompt_label: str, size: int) -> Grid: #prompt_label -> run_mode1에서 "필터 A", "필터 B", "패턴"으로 들어오는 str,size ->int,return grid객체
    """size줄, 공백으로 구분된 숫자 size개를 입력받아 Grid를 생성한다.

    행/열 개수 불일치 또는 숫자 파싱 실패 시 안내 문구를 출력하고 재입력을 유도한다.
    """
    while True:
        print(f"{prompt_label} ({size}줄 입력, 공백 구분)")
        rows: List[List[float]] = [] #로컬변수의 타입힌트,2차원 리스트에 float데이터 넣을건데 지금은 빈리스트
        error = False #flag변수 생성,기본false,입력 오류시 true로 바꿈
        for _ in range(size):#한줄로 입력받으니깐 행 만큼 반복
            tokens = input().strip().split() #행 문자열로 입력받고 앞뒤 공백 제거,숫자 사이 공백 기준으로 문자열을 쪼개서 
            if len(tokens) != size: #token 리스트 길이가 size와 다르면 재입력 요구
                print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
                error = True # 오류 발생으로 flag변수 true로 변경
                break#for문 탈출
            try: # 리스트 수가 맞아도 input값이 숫자가 아닐수도 있으니 일단 실행(try)하고 판단
                values = [float(t) for t in tokens]#tokens 리스트 float으로 변환하여 row 리스트에 추가 
            except ValueError:# input값이 숫자가 아니면 ValueError 발생(a,b,c등), except문 실행
                print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
                error = True #오류발생 flag변수
                break #for문 탈출
            rows.append(values)#tokens 리스트 float으로 변환하여 row 리스트에 추가한 values rows에 추가

        if error: #error = True라면 재입력 요구
            print("다시 입력해주세요.\n")
            continue #while문 처음으로 돌아가서 재입력 요구
        return Grid.from_2d_list(rows) #if문에 안걸린 error = False라면 지금까지 모은 행(row)으로 (객체가 없음으로)Grid 객체를 만들어서 함수를 완전히 종료하며 반환

#출력부분
def run_mode1() -> None: #input_grid, mac_operation, measure_mac_time_ms, print_perf_table 붙여서, "모드 1(사용자 입력)"하나로 조립한 함수
    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")
    filter_a = input_grid("필터 A", 3)#prompt_label: "필터 A", size: 3
    print()
    filter_b = input_grid("필터 B", 3)#prompt_label: "필터 B", size: 3
    print("필터 A, B 저장 완료.")

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")
    pattern = input_grid("패턴", 3)#input_grid 함수로 gird 객체 pattern생성-> prompt_label: "패턴", size: 3 (객체를 만들기전인데 pattern은 grid객체여야함으로 return할떄grid객체로 리턴해서 pattern에 넣음)

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")
    score_a = mac_operation(pattern, filter_a)#pattern과 filter_a를 사용하여 MAC 연산 수행
    score_b = mac_operation(pattern, filter_b)#pattern과 filter_b를 사용하여 MAC 연산 수행
    avg_ms = measure_mac_time_ms(pattern, filter_a, repeats=10) #근데 mac operate에 사이즈 검사있어서 시간 좀 까먹는듯
    #pattern과 filter_a를 사용하여 MAC 연산수행의 평균시간을 repeats=10회 반복하여 측정, filter_b는 값만 다를뿐 연산과정은 동일함으로 노이즈를 제외하면 값이 동일할꺼라 제외

    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/10회): {avg_ms:.3f} ms")

    if abs(score_a - score_b) < EPSILON: #두 점수의 차이가 EPSILON(0.000000001) 보다 작으면 동점(판별불가)
        print("판정: 판정 불가 (|A-B| < 1e-9)")
    else:
        winner = "A" if score_a > score_b else "B" #3항 연산자 사용해서 더 큰쪽 출력
        print(f"판정: {winner}")

    print("\n#----------------------------------------")
    print("# [4] 성능 분석 (3x3)")
    print("#----------------------------------------")
    print_perf_table([(3, avg_ms)]) # print_perf_table 함수에 (3, avg_ms) 튜플을 리스트로 감싸서 전달, 3x3의 평균시간 출력


# ============================================================
# 모드 2: data.json 분석
# ============================================================
def load_filters(data: Dict[str, Any]) -> Dict[int, Dict[str, Grid]]:#필터 불러오기
    """filters 섹션을 로드하여 {size: {'Cross': Grid, 'X': Grid}} 형태로 반환한다."""
    filters: Dict[int, Dict[str, Grid]] = {}# 변수에 타입힌트
    raw_filters = data.get("filters", {})#데이터에서 필터값 가져오기, .get("filters", {})로 접근하여 필터가 없더라도 빈 딕셔너리 받기/ grid클래스의 매소드 get이 아니라 파이썬 딕셔너리 클래스 내장 메소드임
    for key, value in raw_filters.items():#raw_filters의 key-값 순회 
        if not key.startswith("size_"): #키가 size_로 시작하지 않으면 무시하고 넘어가기
            continue
        try:
            size = int(key.split("_")[1])#size_숫자 꼴로 되어있는 key값을 _ 기준으로 나누고 두번째 요소인 숫자를 int형으로 변환
        except (IndexError, ValueError):#size_숫자꼴이 아니라면 무시하고 건너뛰기
            continue
        label_map: Dict[str, Grid] = {}# 변수에 타입힌트 -> size안의 라벨에 대응하는 grid객체를 담을 딕셔너리 새로 만들기
        for raw_label, grid_values in value.items():#value->위에 for문에서 raw_filters.items로 가져온 값, raw_label은 "cross"등의 라벨,grid_value는 2차원리스트
            std_label = normalize_label(raw_label)# 라벨 정규화
            label_map[std_label] = Grid.from_2d_list(grid_values)# label_map의 정규화된 std_label에 cls(size,이차원 리스트) 꼴의 gird 객체 넣기
        filters[size] = label_map#완성된 label_map 대입
    return filters #filter 딕셔너리 반환

#json파일 열기
def run_mode2(json_path: str = "data.json") -> None: #json_path: str = "data.json": 인자를 하나 받는데, 아무것도 안 넘기면 기본값으로 "data.json"이라는 파일 이름을 씀
    print("\n#----------------------------------------")
    print("# [1] 필터 로드") #필터 있는지 확인
    print("#----------------------------------------")

    try:
        with open(json_path, "r", encoding="utf-8") as f: #json_path(기본값 "data.json") 경로의 파일을 "r"(읽기 모드)로, utf-8 인코딩으로 염(한글이 깨지지 않게 인코딩을 명시)
            data = json.load(f) #with ... as f:: 파일을 열고 정상 종료든 에러든 이블럭 끝나면 파이썬이 자동으로 파일을 닫아줌. f라는 이름으로 열린 파일 객체를 사용할 수 있게 됨.
    except FileNotFoundError:#파일을 찾을수 없어서 에러가 뜨면
        print(f"data.json 로드 실패: 파일을 찾을 수 없습니다 ({json_path})")#파일이 없다 출력하고 탈출
        return
    except json.JSONDecodeError as e:# 파일은 존재하는데 json문법오류가 있는 파일이라면
        print(f"data.json 로드 실패: JSON 파싱 오류 ({e})")#로그 e에 저장해서 출력하고 탈출
        return

    filters = load_filters(data)#load_filters 함수에 data넣어서 라벨 정규화된 데이터 대입
    for size in sorted(filters.keys()):#필터의 size를 나타내는 key값 정렬하여 순회
        labels = ", ".join(sorted(filters[size].keys()))#정규화된 라벨 리스트를 쉼표+공백으로 이어붙여서 하나의 문자열로 만들기->["Cross", "X"] → "Cross, X".
        print(f"✓ size_{size:<3}필터 로드 완료 ({labels})")#필터 로드 완료 메세지 출력

    

    print("\n#----------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#----------------------------------------")

    patterns = data.get("patterns", {})#data(JSON) 안에서 patterns라는 실제로 체첨할 값들의 키의 값을 꺼내기.

    total = 0 #테스트 케이스 카운터
    passed = 0#통과한 케이스 카운터
    fail_cases: List[Tuple[str, str]] = []#실패한 케이스를 모아둘 리스트

    for case_id, case in patterns.items():#patterns
        total += 1 #테스트 1회 추가 
        print(f"--- {case_id} ---")#진행중인 케이스 표시

        # 케이스 단위로 예외를 잡아, 스키마 문제가 있어도 프로그램이 중단되지 않게 한다.
        try:
            parts = case_id.split("_")#예를 들어 size_5_1를 "_" 기준으로 쪼갬. 결과는 리스트: ["size", "5", "1"].
            if len(parts) < 2:#쪼겐게 3조각이아니면 오류임
                raise ValueError("케이스 키 형식 오류 (size_N_idx 형태가 아님)")#에러로 판정
            size_n = int(parts[1])#size값 대입 오류뜨면 뒤에 except에서 잡아줌

            if size_n not in filters:# size가 필터에 존재하는 타입이 아니면 오류판정
                raise ValueError(f"size_{size_n} 필터가 존재하지 않음")

            raw_input = case.get("input") #case에서 input 값과  
            raw_expected = case.get("expected") #expected값 꺼내기
            if raw_input is None or raw_expected is None:#둘중 하나라도 없음 오류
                raise ValueError("input/expected 키 누락")

            pattern_size = len(raw_input)# 행 계수 input
            if pattern_size != size_n or any(len(row) != size_n for row in raw_input):#1.행 크기가 안맞거나 2.열의 길이가 행과 다르면 오류 출력
                raise ValueError(
                    f"패턴 크기 불일치 (필터 {size_n}x{size_n}, 패턴 {pattern_size}x?)"
                )

            cross_filter = filters[size_n].get("Cross")#filters[size_n]: 이 크기에 해당하는 라벨맵 딕셔너리(예: {"Cross": Grid객체})를 꺼냄.
            x_filter = filters[size_n].get("X")#"X": Grid객체
            if cross_filter is None or x_filter is None:#없으면 오류
                raise ValueError("Cross/X 필터 라벨 누락")

            pattern_grid = Grid.from_2d_list(raw_input)#문제없는 리스트임으로 grid객체로 만들어서 mac_operation함수에 넘길준비
            score_cross = mac_operation(pattern_grid, cross_filter)#이 패턴이 Cross 모양과 얼마나 일치하는지에 대한 점수.
            score_x = mac_operation(pattern_grid, x_filter)#이 패턴이 X 모양과 얼마나 일치하는지에 대한 점수.

        except ValueError as e:#앞에 있던 try 블록 안에서 ValueError가 발생하면 이 줄로 점프해서 실행됨.
            print(f"FAIL: {e}")#오류 메세지 출력
            fail_cases.append((case_id, str(e)))#실패 케이스에 추가 
            continue #남은코드 실행하지않고 넘어가기 

        if abs(score_cross - score_x) < EPSILON: #판별결과 cross와 x차이가 입실론보다 작다면(근소) 차이 없음으로 판병
            verdict = "UNDECIDED"#판정:모르겠다(동일)
        elif score_cross > score_x:#corss가 크면 corss
            verdict = "Cross"
        else:#아님 x
            verdict = "X"

        expected_std = normalize_label(raw_expected)#표현 정규화해서 넘기기 

        print(f"Cross 점수: {score_cross}")#출력
        print(f"X 점수: {score_x}")#출력

        if verdict == expected_std:#계산으로 나온 판정(verdict)과 JSON에 적힌 정답(expected_std)이 문자열로 정확히 같다면 이 케이스는 맞춤
            print(f"판정: {verdict} | expected: {expected_std} | PASS")
            passed += 1 #패스 숫자 +1
        else: #문자열이 다르면
            reason = "동점 규칙(UNDECIDED)" if verdict == "UNDECIDED" else "판정 불일치" #동점이라 못 맞춘 건지" "아예 잘못 계산한 건지
            print(f"판정: {verdict} | expected: {expected_std} | FAIL ({reason})")
            fail_cases.append((case_id, f"판정={verdict}, expected={expected_std} ({reason})"))

    print("\n#----------------------------------------")
    print("# [3] 성능 분석 (평균/10회)")
    print("#----------------------------------------")
    perf_rows: List[Tuple[int, float]] = []#성능 측정 결과를 모아둘 빈 리스트
    for size in (3, 5, 13, 25):#3은 JSON의 filters에는 없는데 모드1(3x3 사용자 입력)의 성능도 같이 비교하려고 넣음
        if size in filters and "Cross" in filters[size]:#size에 해당하는 필터가 filters 딕셔너리에 실제로 있고,그리고 그 크기의 라벨맵 안에 "Cross"라는 키가 존재하는지도 확인
            filt = filters[size]["Cross"]#그 필터를 그대로 가져다 쓰기, 실제 데이터 기반 성능 측정이라 더 정확함
        else:#3처럼 없는 경우면 JSON 데이터에 의존할 수 없어서 앞서 만든 generate_cross_pattern 함수로 직접 십자가 모양 필터를 만들어서 대신 사용
            filt = generate_cross_pattern(size)
        pat = generate_x_pattern(size)  # 성능 측정 전용 패턴(판정 결과와 무관)
        avg_ms = measure_mac_time_ms(pat, filt, repeats=10)#성능측정하기위한 패턴,결과 상관x
        perf_rows.append((size, avg_ms))
    print_perf_table(perf_rows)#measure_mac_time_ms 함수를 호출해서 pat과 filt로 mac_operation을 repeats(10번) 반복 실행하고, 그 평균 시간 을 계산해서 avg_ms에 저장

    print("\n#----------------------------------------")
    print("# [4] 결과 요약")
    print("#----------------------------------------")
    print(f"총 테스트: {total}개")#케이스 수 출력
    print(f"통과: {passed}개")#pass케이스 출력
    print(f"실패: {total - passed}개")#실패 케이스 출력
    if fail_cases: #실페케이스가 있으면
        print("\n실패 케이스:")#실패케이스 출력
        for cid, reason in fail_cases:#케이스 id랑 실패이유 나누기
            print(f"- {cid}: {reason}") #출력



#===============================================================
#시작화면
#================================================================
print("=== Mini NPU Simulator ===\n")
print("[모드 선택]")
print("1. 사용자 입력 (3x3)")
print("2. data.json 분석")
choice = input("선택: ").strip()

if choice == "1":
    run_mode1()
elif choice == "2":
    run_mode2()
else:
    print("잘못된 선택입니다. 1 또는 2를 입력하세요.")

# ============================================================
# 엔트리 포인트
# ============================================================
'''def main() -> None:
    print("=== Mini NPU Simulator ===\n")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    choice = input("선택: ").strip()        

    if choice == "1":
        run_mode1()
    elif choice == "2":
        run_mode2()
    else:
        print("잘못된 선택입니다. 1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()'''
