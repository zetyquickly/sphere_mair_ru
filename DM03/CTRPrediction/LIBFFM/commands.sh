
# Пропорции фичей
'./utils/count.py tr.csv > feature_proportions.txt'

# Выделяем фичи для бустинга
'converters/parallelizer-train.py -s 10 converters/pre-train.py train train.gbdt.dense train.gbdt.sparse'

# Выделяем фичи для бустинга
'converters/parallelizer-train.py -s 10 converters/pre-train.py test test.gbdt.dense test.gbdt.sparse'
# Генерируем фичи бустингом
'./gbdt -t 30 -s 10 test.gbdt.dense test.gbdt.sparse train.gbdt.dense train.gbdt.sparse test.gbdt.out tr.gbdt.out' 
# Удаляем ненужные файлы
'rm -f test.gbdt.dense test.gbdt.sparse train.gbdt.dense train.gbdt.sparse'

# То же с тестом
'converters/parallelizer-test.py -s 10 converters/pre-test.py train.csv train.gbdt.out train.ffm'

'converters/parallelizer-test.py -s 10 converters/pre-test.py test.csv test.gbdt.out test.ffm'

'rm -f test.gbdt.out train.gbdt.out'

# Применяем FFM
'./ffm-train -k 4 -t 20 -l 0.000005 -s 10 -p te.ffm train.ffm model'

'./ffm-predict test.ffm model test.out'

'./utils/make_submission.py test.out submission.csv'
