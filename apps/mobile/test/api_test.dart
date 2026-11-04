import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:grammarcheck_mobile/api.dart';

void main() {
  test('edits retain emoji offsets and document whitespace', () {
    const text = '🙂 Toi\n\nđi hoc.';
    final edits = [
      {'id': 'a', 'start': 2, 'end': 5, 'source': 'Toi', 'replacement': 'Tôi'},
      {
        'id': 'b',
        'start': 10,
        'end': 13,
        'source': 'hoc',
        'replacement': 'học',
      },
    ];
    expect(applyEdits(text, edits, {'a', 'b'}), '🙂 Tôi\n\nđi học.');
    expect(applyEdits(text, edits, {}), text);
  });
  test('API uses UTF-8, options, and API key', () async {
    final client = MockClient((request) async {
      expect(request.headers['X-API-Key'], 'test');
      expect(jsonDecode(request.body)['text'], 'Tôi đi học.');
      return http.Response(
        jsonEncode({'output': 'Tôi đi học.'}),
        200,
        headers: {'content-type': 'application/json; charset=utf-8'},
      );
    });
    final api = GrammarApi('http://localhost:8000', 'test', client: client);
    final result = await api.check({'text': 'Tôi đi học.'});
    expect(result['output'], 'Tôi đi học.');
    api.close();
  });
  test('API surfaces server errors', () async {
    final api = GrammarApi(
      'http://localhost:8000',
      '',
      client: MockClient(
        (_) async => http.Response('{"detail":"Missing checkpoint"}', 503),
      ),
    );
    await expectLater(api.check({'text': 'test'}), throwsFormatException);
    api.close();
  });
  test('invalid API endpoint is rejected', () {
    expect(() => GrammarApi('https://host/path', ''), throwsFormatException);
    expect(() => GrammarApi('file:///tmp/x', ''), throwsFormatException);
  });
}
