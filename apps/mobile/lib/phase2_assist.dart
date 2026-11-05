import 'dart:convert';

import 'package:flutter/material.dart';

import 'api.dart';

Future<Map<String, String>?> askForContext(BuildContext context) {
  var previous = '', following = '';
  return showDialog<Map<String, String>>(
    context: context,
    builder: (context) => StatefulBuilder(
      builder: (context, update) => AlertDialog(
        title: const Text('Gợi ý theo ngữ cảnh'),
        scrollable: true,
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Có thể thêm câu liền trước hoặc liền sau để bổ sung ngữ cảnh.',
            ),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(labelText: 'Câu trước'),
              minLines: 2,
              maxLines: 3,
              onChanged: (text) => update(() => previous = text),
            ),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(labelText: 'Câu sau'),
              minLines: 2,
              maxLines: 3,
              onChanged: (text) => update(() => following = text),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Đóng'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, {
              'previous': previous.trim(),
              'following': following.trim(),
            }),
            child: const Text('Tìm gợi ý'),
          ),
        ],
      ),
    ),
  );
}

Future<List<dynamic>> fetchContextEdits(
  GrammarApi api,
  Map<String, dynamic> body,
) async {
  final response = await api.client
      .post(
        api.base.resolve('/v1/experimental/phase2'),
        headers: {...api.headers, 'Content-Type': 'application/json'},
        body: jsonEncode(body),
      )
      .timeout(const Duration(minutes: 5));
  return (await api.read(response))['edits'] as List;
}

bool editsOverlap(dynamic a, dynamic b) =>
    (a['start'] == b['start'] && a['end'] == b['end']) ||
    ((a['start'] as int) < (b['end'] as int) &&
        (b['start'] as int) < (a['end'] as int));

bool isContextEdit(dynamic edit) =>
    (edit['id'] as String).startsWith('phase2-');

Set<String> effectiveSelection(List<dynamic> edits, Set<String> selected) {
  final active = edits.where(
    (edit) => isContextEdit(edit) && selected.contains(edit['id']),
  );
  if (active.isEmpty) return selected;
  return {
    for (final edit in edits)
      if (selected.contains(edit['id']) &&
          (isContextEdit(edit) ||
              !active.any((suggestion) => editsOverlap(suggestion, edit))))
        edit['id'] as String,
  };
}

bool canAppendContextEdit(
  dynamic edit,
  List<dynamic> existing,
  Set<String> visibleSelection,
) {
  final start = edit['start'] as int, end = edit['end'] as int;
  final overlapping = existing.where((other) => editsOverlap(edit, other));
  if (overlapping.any(
    (other) =>
        (other['start'] as int) < start ||
        (other['end'] as int) > end ||
        (other['start'] == start &&
            other['end'] == end &&
            other['replacement'] == edit['replacement']),
  )) {
    return false;
  }
  final local = [
    for (final other in overlapping)
      {
        ...Map<String, dynamic>.from(other as Map),
        'start': (other['start'] as int) - start,
        'end': (other['end'] as int) - start,
      },
  ];
  return applyEdits(edit['source'] as String, local, visibleSelection) !=
      edit['replacement'];
}
