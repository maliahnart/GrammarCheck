import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:grammarcheck_mobile/main.dart';

void main() {
  testWidgets('three applications are reachable without an API connection', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(const GrammarCheckApp());
    await tester.pumpAndSettle();
    expect(find.text('Tiếng Việt, rõ hơn.'), findsOneWidget);
    await tester.tap(find.byIcon(Icons.school_outlined));
    await tester.pumpAndSettle();
    expect(find.text('Hiểu tiếng Việt, từng chút.'), findsOneWidget);
    await tester.tap(find.byIcon(Icons.dataset_outlined));
    await tester.pumpAndSettle();
    expect(find.text('Chọn CSV / JSONL'), findsOneWidget);
    expect(find.text('Tên cột văn bản'), findsOneWidget);
  });

  testWidgets('public builds hide connection fields despite a saved endpoint', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues({
      'endpoint': 'http://old.local:8000',
    });
    await tester.pumpWidget(
      const MaterialApp(
        home: Workspace(publicApiBaseUrl: 'http://10.0.2.2:8002'),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byIcon(Icons.tune));
    await tester.pumpAndSettle();
    expect(find.text('Tùy chọn'), findsOneWidget);
    expect(find.text('Địa chỉ API'), findsNothing);
    expect(find.text('API key (nếu có)'), findsNothing);
    expect(find.text('Mô hình'), findsOneWidget);
  });
}
