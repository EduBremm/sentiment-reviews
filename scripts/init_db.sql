-- Cria o banco (executado automaticamente pelo container do postgres)
-- A criação das tabelas é feita pelo SQLAlchemy (src/db/database.py :: init_db).

-- Índices adicionais podem ser criados aqui após as tabelas serem migradas.
DO $$
BEGIN
    RAISE NOTICE 'Banco reviews inicializado. Tabelas serão criadas pela aplicação.';
END $$;
