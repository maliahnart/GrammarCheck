import 'dart:async';
import 'dart:convert';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'api.dart';

void main() => runApp(const GrammarCheckApp());

class GrammarCheckApp extends StatelessWidget {
  const GrammarCheckApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'GrammarCheck · Tiếng Việt',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff285e48)),
      scaffoldBackgroundColor: const Color(0xfff4f5ef),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
      ),
      cardTheme: CardThemeData(
        color: Colors.white,
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    ),
    home: const Workspace(),
  );
}

class Workspace extends StatefulWidget {
  const Workspace({super.key});
  @override
  State<Workspace> createState() => _WorkspaceState();
}

class _WorkspaceState extends State<Workspace> {
  final source = TextEditingController(), terms = TextEditingController();
  final column = TextEditingController(text: 'text');
  final endpoint = TextEditingController(
    text: const String.fromEnvironment(
      'API_BASE_URL',
      defaultValue: 'http://127.0.0.1:8000',
    ),
  );
  final apiKey = TextEditingController();
  String model = 'ml', status = '';
  int page = 0, generation = 0;
  bool preserve = true, abbreviations = false, busy = false, error = false;
  GrammarApi? active;
  Map<String, dynamic>? result, datasetResult;
  Set<String> selected = {};
  List<dynamic> lessons = [];
  final Map<String, int> answers = {};
  Uint8List? datasetBytes;
  String? datasetName;
  static const modes = ['document', 'learn', 'dataset'];

  Map<String, dynamic> get options => {
    'model': model,
    'preserve_style': preserve,
    'normalize_abbrev': abbreviations,
    'protected_terms': terms.text
        .split('\n')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList(),
  };

  @override
  void initState() {
    super.initState();
    restore();
  }

  Future<void> restore() async {
    final preferences = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      endpoint.text = preferences.getString('endpoint') ?? endpoint.text;
      model = preferences.getString('model') ?? 'ml';
      preserve = preferences.getBool('preserve') ?? true;
      abbreviations = preferences.getBool('abbreviations') ?? false;
      terms.text = preferences.getString('terms') ?? '';
    });
  }

  Future<void> run(
    Future<String> Function(GrammarApi api, int ticket) action,
  ) async {
    if (busy) return;
    final ticket = ++generation;
    setState(() {
      busy = true;
      error = false;
      status = 'Đang xử lý…';
    });
    try {
      final api = GrammarApi(endpoint.text, apiKey.text);
      active = api;
      final message = await action(api, ticket);
      if (mounted && ticket == generation) setState(() => status = message);
    } catch (exception) {
      if (mounted && ticket == generation) {
        setState(() {
          error = true;
          status = exception is FormatException
              ? exception.message.toString()
              : exception is TimeoutException
              ? 'Hết thời gian chờ. Hãy chia văn bản nhỏ hơn.'
              : 'Chưa kết nối được API hoặc tệp không đọc được. Kiểm tra kết nối và máy chủ.';
        });
      }
    } finally {
      if (ticket == generation) {
        active?.close();
        active = null;
        if (mounted) setState(() => busy = false);
      }
    }
  }

  void cancel() {
    generation++;
    active?.close();
    active = null;
    setState(() {
      busy = false;
      status = 'Đã hủy chờ kết quả.';
    });
  }

  void invalidate() => setState(() {
    result = null;
    datasetResult = null;
    selected.clear();
  });

  Future<void> check() => run((api, ticket) async {
    if (source.text.trim().isEmpty) {
      throw const FormatException('Vui lòng nhập văn bản.');
    }
    invalidate();
    final data = await api.check({
      'text': source.text,
      'mode': modes[page],
      ...options,
    });
    if (mounted && ticket == generation) {
      setState(() {
        result = data;
        selected = (data['edits'] as List)
            .where((e) => e['accepted'] == true)
            .map((e) => e['id'] as String)
            .toSet();
      });
    }
    return '${(data['edits'] as List).length} đề xuất · ${data['chunks']} đoạn · ${(data['seconds'] as num).toStringAsFixed(1)} giây';
  });

  Future<void> pickText() async {
    try {
      final files = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['txt'],
        withData: true,
      );
      if (files == null || !mounted) return;
      final file = files.files.single;
      if (file.size > 200000 || file.bytes == null) {
        throw const FormatException('Tệp TXT quá lớn hoặc không đọc được.');
      }
      final value = utf8.decode(file.bytes!);
      if (value.runes.length > 30000) {
        throw const FormatException('Văn bản vượt quá 30.000 ký tự.');
      }
      setState(() {
        source.text = value;
        result = null;
        status = 'Đã mở TXT. Bấm kiểm tra để xử lý.';
        error = false;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          status = e is FormatException
              ? e.message.toString()
              : 'Không mở được tệp.';
          error = true;
        });
      }
    }
  }

  Future<void> pickDataset() async {
    try {
      final files = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv', 'jsonl'],
        withData: true,
      );
      if (files == null || !mounted) return;
      final file = files.files.single;
      if (file.size > 2 * 1024 * 1024 || file.bytes == null) {
        throw const FormatException('Tệp cần nhỏ hơn 2 MB.');
      }
      setState(() {
        datasetName = file.name;
        datasetBytes = file.bytes;
        datasetResult = null;
        status = '';
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          status = e is FormatException
              ? e.message.toString()
              : 'Không mở được tệp.';
          error = true;
        });
      }
    }
  }

  Future<void> share(String value, {String? filename, String? mime}) async {
    try {
      final box = context.findRenderObject() as RenderBox?;
      await SharePlus.instance.share(
        ShareParams(
          text: filename == null ? value : null,
          files: filename == null
              ? null
              : [XFile.fromData(utf8.encode(value), mimeType: mime)],
          fileNameOverrides: filename == null ? null : [filename],
          sharePositionOrigin: box == null
              ? null
              : box.localToGlobal(Offset.zero) & box.size,
        ),
      );
    } catch (_) {
      if (mounted) {
        setState(() {
          status = 'Chưa chia sẻ được. Bạn có thể sao chép kết quả.';
          error = true;
        });
      }
    }
  }

  Future<void> settings() async {
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (sheetContext) => StatefulBuilder(
        builder: (context, update) => Padding(
          padding: EdgeInsets.fromLTRB(
            24,
            24,
            24,
            MediaQuery.viewInsetsOf(context).bottom + 24,
          ),
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Kết nối & tùy chọn',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: endpoint,
                  decoration: const InputDecoration(
                    labelText: 'Địa chỉ API',
                    helperText: 'Máy thật: dùng IP LAN hoặc HTTPS của máy chủ.',
                  ),
                  keyboardType: TextInputType.url,
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: apiKey,
                  obscureText: true,
                  autocorrect: false,
                  enableSuggestions: false,
                  decoration: const InputDecoration(
                    labelText: 'API key (nếu có)',
                    helperText: 'Chỉ giữ trong phiên mở ứng dụng.',
                  ),
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<String>(
                  initialValue: model,
                  decoration: const InputDecoration(labelText: 'Mô hình'),
                  items: const [
                    DropdownMenuItem(value: 'ml', child: Text('ML Baseline')),
                    DropdownMenuItem(
                      value: 'bilstm',
                      child: Text('BiLSTM + Attention'),
                    ),
                    DropdownMenuItem(
                      value: 'transformer',
                      child: Text('Transformer ViT5'),
                    ),
                  ],
                  onChanged: (value) => update(() => model = value!),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Bảo toàn văn phong'),
                  value: preserve,
                  onChanged: (value) => update(() => preserve = value),
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Mở rộng viết tắt'),
                  value: abbreviations,
                  onChanged: (value) => update(() => abbreviations = value),
                ),
                TextField(
                  controller: terms,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Thuật ngữ giữ nguyên',
                    helperText: 'Mỗi dòng một tên riêng hoặc thuật ngữ.',
                  ),
                ),
                const SizedBox(height: 16),
                FilledButton(
                  onPressed: () => Navigator.pop(sheetContext),
                  child: const Text('Lưu & kiểm tra kết nối'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
    if (!mounted) return;
    invalidate();
    await run((api, ticket) async {
      await api.get('/health');
      final preferences = await SharedPreferences.getInstance();
      await preferences.setString('endpoint', endpoint.text);
      await preferences.setString('model', model);
      await preferences.setBool('preserve', preserve);
      await preferences.setBool('abbreviations', abbreviations);
      await preferences.setString('terms', terms.text);
      return 'Đã lưu kết nối API.';
    });
  }

  Widget editor() => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Row(
        children: [
          const Expanded(
            child: Text(
              'VĂN BẢN CỦA BẠN',
              style: TextStyle(fontSize: 11, letterSpacing: 1.5),
            ),
          ),
          TextButton.icon(
            onPressed: busy ? null : pickText,
            icon: const Icon(Icons.upload_file, size: 18),
            label: const Text('Mở TXT'),
          ),
        ],
      ),
      TextField(
        controller: source,
        enabled: !busy,
        minLines: 6,
        maxLines: 14,
        maxLength: 30000,
        autocorrect: false,
        decoration: const InputDecoration(
          hintText: 'Nhập câu, đoạn văn hoặc ví dụ bạn muốn tìm hiểu…',
        ),
        onChanged: (_) => invalidate(),
      ),
      const Text(
        'Chỉ gửi văn bản đến API khi bạn bấm kiểm tra.',
        style: TextStyle(fontSize: 12, color: Colors.black54),
      ),
      const SizedBox(height: 12),
      FilledButton.icon(
        onPressed: busy ? null : check,
        icon: const Icon(Icons.spellcheck),
        label: const Text('Kiểm tra văn bản'),
      ),
    ],
  );

  Widget resultCard() {
    final data = result!;
    final output = applyEdits(
      data['input'] as String,
      data['edits'] as List,
      selected,
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 24),
        Row(
          children: [
            Text(
              'Bản sửa đang chọn',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const Spacer(),
            IconButton(
              tooltip: 'Sao chép',
              onPressed: () async {
                await Clipboard.setData(ClipboardData(text: output));
                if (mounted) {
                  ScaffoldMessenger.of(
                    context,
                  ).showSnackBar(const SnackBar(content: Text('Đã sao chép.')));
                }
              },
              icon: const Icon(Icons.copy),
            ),
            IconButton(
              tooltip: 'Chia sẻ bản sửa',
              onPressed: () => share(output),
              icon: const Icon(Icons.ios_share),
            ),
          ],
        ),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: SelectableText(output, style: const TextStyle(height: 1.65)),
          ),
        ),
        const SizedBox(height: 12),
        const Text('Duyệt thay đổi · Bật/tắt từng đề xuất'),
        for (final dynamic item in data['edits'] as List)
          Card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                CheckboxListTile(
                  value: selected.contains(item['id']),
                  title: Text(
                    '${(item['source'] as String).isEmpty ? '∅' : item['source']} → ${(item['replacement'] as String).isEmpty ? '∅' : item['replacement']}',
                  ),
                  subtitle: Text(
                    '${item['explanation']}${item['reason'] == '' ? '' : '\n${item['reason']}'}',
                  ),
                  onChanged: (value) => setState(() {
                    value!
                        ? selected.add(item['id'] as String)
                        : selected.remove(item['id']);
                  }),
                ),
                if (page == 1 && item['lesson_id'] != null)
                  TextButton(
                    onPressed: busy ? null : loadLessons,
                    child: const Text('Khám phá bài học liên quan'),
                  ),
              ],
            ),
          ),
      ],
    );
  }

  Future<void> loadLessons() => run((api, ticket) async {
    final data = await api.get('/v1/lessons');
    if (mounted && ticket == generation) {
      setState(() => lessons = data['lessons'] as List);
    }
    return 'Đã mở thư viện học tiếng Việt.';
  });

  Widget learning() => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      const SizedBox(height: 24),
      Text(
        'Khám phá tiếng Việt',
        style: Theme.of(context).textTheme.titleLarge,
      ),
      const Text('Chính tả · Từ vựng · Ngữ pháp · Giao tiếp'),
      const SizedBox(height: 12),
      OutlinedButton.icon(
        onPressed: busy ? null : loadLessons,
        icon: const Icon(Icons.menu_book),
        label: const Text('Mở bài học & câu hỏi'),
      ),
      for (final dynamic lesson in lessons)
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  lesson['category'] as String,
                  style: const TextStyle(fontSize: 12, color: Colors.black54),
                ),
                const SizedBox(height: 6),
                Text(
                  lesson['title'] as String,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 10),
                Text(
                  lesson['explanation'] as String,
                  style: const TextStyle(height: 1.6),
                ),
                for (final example in lesson['examples'] as List)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(example as String),
                  ),
                const Divider(height: 28),
                Text(
                  lesson['question'] as String,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                for (
                  var index = 0;
                  index < (lesson['choices'] as List).length;
                  index++
                )
                  OutlinedButton(
                    onPressed: () =>
                        setState(() => answers[lesson['id'] as String] = index),
                    child: Text(lesson['choices'][index] as String),
                  ),
                if (answers.containsKey(lesson['id']))
                  Text(
                    '${answers[lesson['id']] == lesson['answer'] ? 'Đúng.' : 'Chưa đúng.'} ${lesson['feedback']}',
                    style: const TextStyle(height: 1.6),
                  ),
              ],
            ),
          ),
        ),
    ],
  );

  Widget dataset() => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      const Text(
        'CSV/JSONL · UTF-8 · tối đa 300 bản ghi, 2 MB và 100.000 ký tự. Giữ bản gốc, thêm bản chuẩn hóa và nhật ký sửa.',
        style: TextStyle(height: 1.6),
      ),
      const SizedBox(height: 18),
      TextField(
        controller: column,
        enabled: !busy,
        decoration: const InputDecoration(labelText: 'Tên cột văn bản'),
        onChanged: (_) => setState(() => datasetResult = null),
      ),
      const SizedBox(height: 12),
      OutlinedButton.icon(
        onPressed: busy ? null : pickDataset,
        icon: const Icon(Icons.folder_open),
        label: Text(datasetName ?? 'Chọn CSV / JSONL'),
      ),
      const SizedBox(height: 12),
      FilledButton.icon(
        onPressed: busy || datasetBytes == null
            ? null
            : () => run((api, ticket) async {
                setState(() => datasetResult = null);
                final data = await api.clean(
                  datasetBytes!,
                  datasetName!,
                  column.text,
                  options,
                );
                if (mounted && ticket == generation) {
                  setState(() => datasetResult = data);
                }
                return 'Đã xử lý ${data['rows']} bản ghi; ${data['changed_rows']} bản ghi thay đổi.';
              }),
        icon: const Icon(Icons.auto_fix_high),
        label: const Text('Chuẩn hóa dữ liệu'),
      ),
      if (datasetResult != null) ...[
        const SizedBox(height: 16),
        OutlinedButton.icon(
          onPressed: () => share(
            datasetResult!['content'] as String,
            filename: datasetResult!['filename'] as String,
            mime: datasetResult!['content_type'] as String,
          ),
          icon: const Icon(Icons.ios_share),
          label: const Text('Lưu / chia sẻ tệp kết quả'),
        ),
        TextButton(
          onPressed: () => Clipboard.setData(
            ClipboardData(text: datasetResult!['content'] as String),
          ),
          child: const Text('Sao chép dữ liệu kết quả'),
        ),
      ],
    ],
  );

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      backgroundColor: const Color(0xfff4f5ef),
      title: const Text(
        'GrammarCheck',
        style: TextStyle(fontWeight: FontWeight.w700),
      ),
      actions: [
        IconButton(
          tooltip: 'Kết nối & tùy chọn',
          onPressed: busy ? null : settings,
          icon: const Icon(Icons.tune),
        ),
      ],
    ),
    body: SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 760),
          child: ListView(
            padding: const EdgeInsets.all(22),
            children: [
              Text(
                [
                  'Tiếng Việt, rõ hơn.',
                  'Hiểu tiếng Việt, từng chút.',
                  'Dữ liệu sạch, có kiểm soát.',
                ][page],
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                [
                  'Rà soát tài liệu, giữ cách diễn đạt của bạn.',
                  'Tìm hiểu từ ngữ, cấu trúc câu và hoàn cảnh giao tiếp.',
                  'Chuẩn hóa hàng loạt, luôn giữ lại văn bản gốc.',
                ][page],
                style: const TextStyle(color: Colors.black54, height: 1.5),
              ),
              const SizedBox(height: 20),
              if (page == 2) dataset() else editor(),
              if (busy) ...[
                const SizedBox(height: 16),
                const LinearProgressIndicator(),
                TextButton(onPressed: cancel, child: const Text('Hủy chờ')),
              ],
              if (status.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  child: Text(
                    status,
                    style: TextStyle(
                      color: error
                          ? Colors.red.shade800
                          : const Color(0xff285e48),
                      height: 1.5,
                    ),
                  ),
                ),
              if (result != null && page != 2) resultCard(),
              if (page == 1) learning(),
              const SizedBox(height: 30),
              const Divider(),
              const Text(
                'Văn bản được xử lý tại máy chủ bạn cấu hình. Bản này cần kết nối API; hãy đọc lại các gợi ý trước khi sử dụng.',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.black54,
                  height: 1.6,
                ),
              ),
            ],
          ),
        ),
      ),
    ),
    bottomNavigationBar: NavigationBar(
      selectedIndex: page,
      onDestinationSelected: busy
          ? null
          : (value) => setState(() {
              page = value;
              status = '';
            }),
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.description_outlined),
          selectedIcon: Icon(Icons.description),
          label: 'Tài liệu',
        ),
        NavigationDestination(
          icon: Icon(Icons.school_outlined),
          selectedIcon: Icon(Icons.school),
          label: 'Học tiếng Việt',
        ),
        NavigationDestination(
          icon: Icon(Icons.dataset_outlined),
          selectedIcon: Icon(Icons.dataset),
          label: 'Dữ liệu NLP',
        ),
      ],
    ),
  );

  @override
  void dispose() {
    active?.close();
    for (final controller in [source, terms, column, endpoint, apiKey]) {
      controller.dispose();
    }
    super.dispose();
  }
}
