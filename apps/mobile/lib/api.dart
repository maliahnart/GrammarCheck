import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class GrammarApi {
  GrammarApi(String baseUrl, this.key, {http.Client? client})
    : base = Uri.parse(baseUrl.trim()),
      client = client ?? http.Client() {
    if (!['http', 'https'].contains(base.scheme) ||
        base.host.isEmpty ||
        base.userInfo.isNotEmpty ||
        base.hasQuery ||
        base.hasFragment ||
        !['', '/'].contains(base.path)) {
      throw const FormatException(
        'API cần có dạng https://host hoặc http://IP:8000.',
      );
    }
    if (kReleaseMode && base.scheme != 'https') {
      throw const FormatException('Bản phát hành cần kết nối API qua HTTPS.');
    }
  }
  final Uri base;
  final String key;
  final http.Client client;
  Map<String, String> get headers => {if (key.isNotEmpty) 'X-API-Key': key};

  Future<Map<String, dynamic>> read(http.Response response) async {
    final data =
        jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    if (response.statusCode >= 400) {
      throw FormatException(
        data['detail'] is String
            ? data['detail'] as String
            : 'Văn bản hoặc cấu hình không hợp lệ.',
      );
    }
    return data;
  }

  Future<Map<String, dynamic>> get(String path) async => read(
    await client
        .get(base.resolve(path), headers: headers)
        .timeout(const Duration(seconds: 20)),
  );

  Future<Map<String, dynamic>> check(Map<String, dynamic> body) async => read(
    await client
        .post(
          base.resolve('/v1/check'),
          headers: {...headers, 'Content-Type': 'application/json'},
          body: jsonEncode(body),
        )
        .timeout(const Duration(minutes: 5)),
  );

  Future<Map<String, dynamic>> clean(
    Uint8List bytes,
    String filename,
    String column,
    Map<String, dynamic> options,
  ) async {
    final request = http.MultipartRequest(
      'POST',
      base.resolve('/v1/datasets/clean'),
    );
    request.headers.addAll(headers);
    request.fields.addAll({
      'column': column,
      for (final entry in options.entries)
        entry.key: entry.value is List
            ? jsonEncode(entry.value)
            : '${entry.value}',
    });
    request.files.add(
      http.MultipartFile.fromBytes('file', bytes, filename: filename),
    );
    final stream = await client
        .send(request)
        .timeout(const Duration(minutes: 5));
    return read(
      await http.Response.fromStream(
        stream,
      ).timeout(const Duration(minutes: 5)),
    );
  }

  void close() => client.close();
}

String applyEdits(String source, List<dynamic> edits, Set<String> selected) {
  final points = source.runes.toList();
  final ordered = edits.map((e) => Map<String, dynamic>.from(e as Map)).toList()
    ..sort((a, b) {
      final order = (a['start'] as int).compareTo(b['start'] as int);
      return order == 0 ? (a['end'] as int).compareTo(b['end'] as int) : order;
    });
  final out = StringBuffer();
  var cursor = 0;
  for (final edit in ordered) {
    if (!selected.contains(edit['id'])) continue;
    final start = edit['start'] as int, end = edit['end'] as int;
    if (start < cursor ||
        String.fromCharCodes(points.sublist(start, end)) != edit['source']) {
      throw const FormatException(
        'Phép sửa không khớp văn bản. Hãy kiểm tra lại.',
      );
    }
    out.write(String.fromCharCodes(points.sublist(cursor, start)));
    out.write(edit['replacement']);
    cursor = end;
  }
  out.write(String.fromCharCodes(points.sublist(cursor)));
  return out.toString();
}
