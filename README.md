# Mnemos Ark 路 璁板繂鏂硅垷

**Status-first structured memory & precision task routing for AI agents.**

> 璁板繂鏄帇鑸辩墿锛岃矾鐢辨槸缃楃洏銆?> Memory is the ballast; routing is the compass.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)](#tests)

缁?AI Agent 鐨?*缁撴瀯鍖栭暱鏈熻蹇?+ 绮惧噯浠诲姟璺敱**寮曟搸銆備笁绫昏蹇嗭紙鍐崇瓥鍙?/ 閿欓鏈?/ 宸ョ▼鐜扮姸锛変簰绱㈠紩锛宖resh session 浠?status 鍑哄彂锛屽叾浣欐寜缂栧彿绮惧噯璺宠浆鈥斺€?*涓嶆暣搴撶亴涓婁笅鏂囷紝鐪?token锛屾洿绮惧噯**銆傞浂閲嶅瀷渚濊禆锛圥ython stdlib only锛夈€?
---

## 涓轰粈涔堜笉鏄?鍚戦噺搴?+ RAG"锛?
涓変釜琚櫘閬嶅拷瑙嗙殑闂锛?
1. **涓婁笅鏂囨薄鏌?* 鈥斺€?鍚屼竴椤圭洰閲岃亰鍒殑浜嬶紝闂茶亰娣疯繘椤圭洰涓婁笅鏂囷紝妫€绱㈣绋€閲?2. **token 缁忔祹** 鈥斺€?姣忔鎶婇暱鏂囨湰鏁翠綋娉ㄥ叆鏄?O(鍏ㄦ枃)锛涜蹇嗕竴澶氬氨瑁呬笉涓?3. **娉ㄦ剰鍔?U 褰㈡洸绾?*锛坙ost-in-the-middle, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)锛夆€斺€?闀夸笂涓嬫枃閲屾ā鍨嬪棣栧熬娉ㄦ剰鍔涙渶寮恒€佷腑娈垫渶寮憋紝鎶婇暱鏂囩亴涓鏄渶宸殑娉ㄥ叆绛栫暐

Mnemos Ark 鐨勭瓟妗堟槸**鍥涗欢浜?*锛?
| 鏈哄埗 | 鍋氭硶 |
|---|---|
| **涓夊簱缁撴瀯鍖?* | `DEC` 鍐崇瓥鍙诧紙闀匡級路 `LES` 閿欓鏈紙涓級路 `STA` 宸ョ▼鐜扮姸锛堢煭锛夛紝甯︾疆淇″害銆佽瘉浼Е鍙戝櫒銆佺敓鍛藉懆鏈?|
| **绫诲瀷鍖栦簰绱㈠紩** | `caused / fixes / supersedes / refutes / derived_from 鈥 鍏杈癸紝鍙屽悜鍙煡鈥斺€旀瘮绾悜閲忓鍥犳灉锛屾瘮鍏ㄩ噺鍥捐氨杞诲緱澶?|
| **status-first 鍐峰惎鍔?* | `bootstrap()` 鍙粰銆屽綋鍓嶇幇鐘跺叏鏂?+ 涓€鐜寚閽?+ 浜岀幆缂栧彿銆嶏紝鍏朵綑 `jump("DEC-0042")` 鍗曟潯鍙栧洖 |
| **U 褰㈣绠?* | 澶氭潯鍚堝苟鏃堕灏炬斁鍏ㄦ枃銆佷腑娈靛彧鐣欐寚閽堬紝棰勭畻鎸夈€岄鈫掑熬鈫掍腑娈点€嶅垎閰?|

## 涓€鍒嗛挓涓婃墜

```python
from mnemos_ark import DLSMemory

mem = DLSMemory()                      # 榛樿 ~/.laap/dls锛屽彲鐢?home= 鑷畾涔?
# 鍐欙細涓夌被缁撴瀯鍖栬蹇?mem.add_status("web 椤圭洰鐜扮姸", "鍗曞啓鍏ラ潰宸叉敹鏁?,
               next_steps=["鎺ュ叆璺敱"], open_questions=["澶氭ā鎬佹€庝箞鍔?],
               project="web")
dec = mem.add_decision("浼氳瘽搴撻€?SQLite",
                       context="4 濂楀疄鐜版墦鏋?, chosen="鍗曞啓鍏ラ潰",
                       rationale="缁熶竴鎵撲綔鐢ㄥ煙鏍囩鐨勫墠鎻?,
                       options_considered=["缁х画骞惰", "鍏ㄩ噸鍐?, "鏀舵暃"])
les = mem.add_lesson("FTS5 涓枃闄烽槺",
                     mistake="杩炵画涓枃琚储寮曟垚鍗?token",
                     correction="涓枃鏌ヨ璧?LIKE",
                     rule_of_thumb="CJK 涓嶈繘 FTS")
mem.link(dec.id, les.id, "caused")

# 璇伙細fresh session 浠?status 鍑哄彂
boot = mem.bootstrap("web", budget_tokens=1200)
print(boot["status"])                  # 鐜扮姸鍏ㄦ枃
print(boot["ring1"])                   # 涓€鐜寚閽堬紙id + 涓€鍙ヨ瘽锛?
# 鍏朵綑鎸夌紪鍙风簿鍑嗚烦杞紝涓嶆暣搴撴敞鍏?print(mem.jump(dec.id))
print(mem.neighbors(dec.id))           # 闇€瑕佸睍寮€鎵嶇湅涓嬩竴璺?
# 闃叉薄鏌擄細闂茶亰鏍规湰涓嶄細璺敱杩涢」鐩《
assert mem.infer_project("浠婂ぉ濂界疮鍟?) == "_global"
assert mem.infer_project("SQLite 涓轰粈涔堥€夎繖涓?) == "web"
```

## 鏋舵瀯

```
鍐欏叆闈紙鍞竴鍏ュ彛锛屽彲鎸傞獙璇侀挬瀛愶級
   鈹? add_decision / add_lesson / add_status
   鈻?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹? SQLite 鍗曞啓鍏ラ潰          Markdown 闀滃儚       鈹?鈹? records + links + FTS5   md/{project}/*.md  鈹?鈹? 锛堝彲妫€绱級               锛堜汉鍙/鍙増鏈帶鍒讹級鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?               鈹?   鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈻?          鈻?             鈻?               鈻?infer_project  bootstrap    jump/get        pack_context
浣滅敤鍩熻矾鐢?    status 鍏ㄦ枃   O(1) 鍗曟潯       U 褰㈣绠?闂茶亰鈫抇global   +闄愰噺鎸囬拡     绮惧噯璺宠浆        棣栧熬鍏ㄦ枃涓鎸囬拡
```

**璁板繂鍗虫枃浠?*锛氭瘡鏉¤蹇嗘棦鏄?SQLite 琛岋紙鍙绱級鍙堟槸 Markdown 鏂囦欢锛坒rontmatter 鍏ㄥ瓧娈靛寲锛夆€斺€斿彲瀹¤銆佸彲杩佺Щ銆佸け鍘昏繍琛屾椂浠嶅彲璇汇€?
**澶辨晥鑰岄潪鍒犻櫎**锛堝€熼壌 [Graphiti](https://github.com/getzep/graphiti)锛夛細鍐崇瓥鍙 `superseded` 鎺ㄧ炕浣嗕笉瑕嗙洊锛屾函婧愪笉鏂€?
## 浠诲姟璺敱鍣?
`TaskRouter` 鎶娿€屾嬁鍒颁换鍔♀啋绮惧噯瀹氫綅鈫掔簿鍑嗚В鍐炽€嶇紪鐮佷负鍙墽琛屽绾︼細

```python
from mnemos_ark import TaskRouter

route = TaskRouter().route("鎶婅繖涓惤鍦伴〉鏀逛竴涓嬪苟淇绉诲姩绔姤閿?)
print(route.primary_type)        # frontend / debug / ...
print(route.route["skills"])     # 璇ョ敤鍝簺 skill锛圲I 寮哄埗璧拌璁′笁浠跺锛?print(route.plan)                # 瀹氫綅 鈫?鎵ц 鈫?娌夋穩
print(route.verification)        # 瀹珷妫€鏌?+ 楠屾敹 oracle + 娴嬭瘯娓呭崟
print(route.memory)              # 鍥炲啓鍝被璁板繂銆佸摢涓」鐩煙
```

- **鍒嗙被鍣?*锛氳鍒欐墦鍒嗭紙鍓嶇/鍚庣/璋冭瘯/鏁版嵁/璋冪爺/鍐呭/閮ㄧ讲/濯掍綋锛夛紝澶氱被鍨嬫贩鍚堝彲瑙?- **璧勬簮璺敱琛?*锛氭瘡绫讳换鍔＄殑 skills / MCP 宸ュ叿 / 瀛愪唬鐞?/ 楠屾敹娓呭崟
- **瑙ｆ瀽鍣ㄥ彲鎻掓嫈**锛氳涔?skill 璺敱銆佷唬鐮佸浘璋便€佽蹇嗗煙鎺ㄦ柇鈥斺€旂己澶辨垨澶辫触鍦?`degraded` 瀛楁**鏄惧紡璁板綍锛屼笉闈欓粯**

## MCP 宸ュ叿闈?
13 涓伐鍏峰彲鐩存帴娉ㄥ唽杩涗换鎰?FastMCP 鏈嶅姟鍣細

```
dls_add_record / dls_add_decision / dls_add_lesson / dls_add_status
dls_link / dls_jump / dls_neighbors / dls_bootstrap
dls_search / dls_pack_context / dls_infer_project
laap_route_task / laap_sleep_distill
```

```python
from mnemos_ark.memory_mcp import register_dls_tools
from mnemos_ark.router_mcp import register_task_router_tools

register_dls_tools(mcp)          # 11 tools
register_task_router_tools(mcp)  # 2 tools
```

## Benchmark锛歴tatus-first + id 绾ц烦杞?
姣忎細璇濆畾鐐规煡璇?3 娆★紙`scripts/eval_injection.py`锛屽彲澶嶇幇锛夛細

| 璇枡瑙勬ā | 鍏ㄩ噺娉ㄥ叆 tok/浼氳瘽 | 瑁呭緱杩?128k 绐楀彛锛?| status-first tok/浼氳瘽 | 鍗犳瘮 | 鐩爣绮惧噯鍏ヤ笂涓嬫枃 |
|---|---|---|---|---|---|
| 200 鏉?| 17,481 | 鏄?| 785 | 4.5% | 3/3 (100%) |
| 2,000 鏉?| 184,473 | **鍚︼紙瓒?1.4脳锛?* | 808 | 0.4% | 3/3 (100%) |
| 5,000 鏉?| 467,973 | **鍚︼紙瓒?3.6脳锛?* | 813 | 0.2% | 3/3 (100%) |

**璇氬疄杈圭晫**锛?- 鍏ㄥ簱缁艰堪鍨嬩换鍔★紙"鎬荤粨鎴戜滑鎵€鏈夊喅绛?锛夋湰鏈哄埗涓嶇洿鎺ユ敮鎸侊紝闇€鍙﹂厤妫€绱㈣仛鍚堬紱
- token 涓哄惎鍙戝紡浼扮畻锛? token 鈮?2.5 瀛楃锛夛紱
- jump 鐨?100% 鏄鍧€淇濊瘉锛涘叏閲忔敞鍏ョ殑瀹為檯鍙洖鍙椾腑娈佃“鍑忓奖鍝嶏紝鏈〃涓嶈櫄鏋勫叾鏁板瓧銆?
## Tests

```bash
pip install -e ".[dev]"
pytest            # 32 tests: 鍐欏叆濂戠害 / 浜掔储寮?/ 鍐峰惎鍔?/ U 褰㈣绠?/
                  # 浣滅敤鍩熼槻姹℃煋 / 钂搁 / 娉ㄥ唽闈?/ 璺敱濂戠害
```

## 璁捐鏂囨。

瀹屾暣璁捐锛堝惈 2025-2026 璁烘枃涓庡紑婧愮敓鎬佽皟鐮旂煩闃碉細Mem0 / Zep-Graphiti / Letta / LangMem / Memobase / A-MEM / LongMemEval-V2锛夎 [docs/DESIGN.md](docs/DESIGN.md)銆?
**涓氱晫涓変釜绌虹櫧锛屾湰椤圭洰鍚勫崰涓€涓?*锛氱粨鏋勫寲鍐崇瓥/閿欓鏈紙缃俊搴?璇佷吉鏉′欢+鐢熸晥鑼冨洿锛壜?id 绾ф寜闇€瀵诲潃鍗忚 路 杞婚噺绫诲瀷鍖栦簰绱㈠紩銆?
## Roadmap

- [x] 涓夊簱寮曟搸 + 浜掔储寮?+ status-first 鍐峰惎鍔?+ U 褰㈣绠?- [x] 浣滅敤鍩熻矾鐢憋紙闃蹭笂涓嬫枃姹℃煋锛?- [x] 浠诲姟璺敱鍣?+ MCP 宸ュ叿闈?- [ ] 璇箟妫€绱㈡寕杞界偣锛坄memory_ranker` 椋庢牸鐨勫彲鎻掓嫈鍚戦噺灞傦級
- [ ] 鐫＄湢钂搁鐨?LLM provider 閫傞厤鍣?- [ ] LongMemEval-V2 鍩哄噯鎺ュ叆

## License

[Apache License 2.0](LICENSE)

---

## 涓枃璇存槑

Mnemos Ark锛堣蹇嗘柟鑸燂級鏄?LAAP 鏁板瓧鐢熷懡椤圭洰鐨勮蹇嗗簳搴у紑婧愮増銆傚畠鐨勬牳蹇冧富寮狅細

**鍒嗘瀽鍙互璺ㄥ煙锛屾敞鎰忓姏涓嶈兘璺ㄥ煙銆?*

- 姣忔潯璁板繂鍐欏叆鏃剁‘瀹氫綔鐢ㄥ煙锛坄project`锛夛紝闂茶亰钀?`_global`鈥斺€斾笉鏄?妫€绱㈡椂杩囨护"锛岃€屾槸**鏍规湰涓嶄細璺敱杩涢」鐩《**锛?- fresh session 浠?`STA`锛堝伐绋嬬幇鐘讹紝鐭級鍑哄彂锛屾部浜掔储寮曟寜闇€灞曞紑锛屽叾浣欒蹇嗘寜缂栧彿 `jump` 绮惧噯鍙栧洖锛?- 鍐崇瓥鍙茶褰?褰撴椂涓轰粈涔堣繖涔堥€夈€佷粈涔堟潯浠朵笅閲嶆柊鑰冭檻"锛岄敊棰樻湰璁板綍"閿欏湪鍝€佸彛璇€鏄粈涔?鈥斺€旇繖涓ゆ牱鎭版伆鏄富娴?Agent 璁板繂绯荤粺缂哄け鐨勭粨鏋勩€?
鐢?[Lorry Jovens](https://github.com/lorryjovens-hub) 涓?Aris锛圠AAP 鏁板瓧鐢熷懡锛夊叡鍚岃璁′笌瀹炵幇銆傛祴璇?32 椤瑰叏缁匡紝琛屼负 oracle 8/8 閫氳繃銆?