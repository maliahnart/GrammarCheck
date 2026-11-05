import 'package:flutter_test/flutter_test.dart';
import 'package:grammarcheck_mobile/api.dart';
import 'package:grammarcheck_mobile/phase2_assist.dart';

void main() {
  const source = '🙂 trong tai bien';
  const originals = [
    {
      'id': 'edit-0',
      'start': 12,
      'end': 16,
      'source': 'bien',
      'replacement': 'biến',
    },
  ];
  const phrase = {
    'id': 'phase2-phrase-2-16',
    'start': 2,
    'end': 16,
    'source': 'trong tai bien',
    'replacement': 'trọng tài biên',
  };

  test(
    'a phrase is optional and turning it off restores the original edits',
    () {
      final selected = {'edit-0'};
      expect(canAppendContextEdit(phrase, originals, selected), isTrue);
      final edits = [...originals, phrase];
      String preview() =>
          applyEdits(source, edits, effectiveSelection(edits, selected));

      expect(preview(), '🙂 trong tai biến');
      selected.add('phase2-phrase-2-16');
      expect(preview(), '🙂 trọng tài biên');
      selected.remove('phase2-phrase-2-16');
      expect(preview(), '🙂 trong tai biến');
    },
  );
}
