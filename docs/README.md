# 📚 Documentação RAG Demo

Bem-vindo à documentação completa do projeto **RAG Demo** - Sistema profissional de Retrieval-Augmented Generation com LangChain, ChromaDB e Ollama.

## 📖 Documentos Principais

### Essenciais
- **[FAQ - Perguntas Frequentes](FAQ.md)** - Respostas para dúvidas comuns sobre modelos, embeddings, tokenização e troubleshooting
- **[Arquitetura do Sistema](ARCHITECTURE.md)** - Estrutura técnica, componentes e fluxo de dados
- **[Changelog](CHANGELOG.md)** - Histórico completo de mudanças e versões

### Guias Práticos

#### 🚀 Primeiros Passos
- **[Início Rápido](guides/quickstart.md)** - Configure e execute o projeto em minutos
- **[Resolução de Problemas](guides/troubleshooting.md)** - Soluções para erros comuns

#### 🔧 Configuração Avançada
- **[Guia de Modelos LLM](guides/models.md)** - Como escolher e configurar modelos Ollama
- **[Guia de Embeddings](guides/embeddings.md)** - Entenda e otimize os embeddings

## 🎯 Por Onde Começar?

### Se você é novo no projeto:
1. Leia o [README principal](../README.md) para visão geral
2. Siga o [Início Rápido](guides/quickstart.md) para instalação
3. Consulte o [FAQ](FAQ.md) para dúvidas comuns

### Se você quer entender a fundo:
1. Estude a [Arquitetura](ARCHITECTURE.md)
2. Explore os [Guias de Configuração](guides/)
3. Veja o [Changelog](CHANGELOG.md) para evolução do projeto

### Se você está com problemas:
1. Consulte o [Troubleshooting](guides/troubleshooting.md)
2. Verifique o [FAQ](FAQ.md)
3. Abra uma [issue no GitHub](https://github.com/patrickmcruz/rag-demo/issues)

## 📂 Estrutura da Documentação

```
docs/
├── README.md                    # Este arquivo - índice da documentação
├── FAQ.md                       # Perguntas frequentes
├── ARCHITECTURE.md              # Arquitetura do sistema
├── CHANGELOG.md                 # Histórico de mudanças
└── guides/                      # Guias detalhados
    ├── quickstart.md            # Início rápido
    ├── embeddings.md            # Guia de embeddings
    ├── models.md                # Guia de modelos LLM
    └── troubleshooting.md       # Resolução de problemas
```

## 🛠️ Recursos do Projeto

### Código-fonte
- **[src/ingest.py](../src/ingest.py)** - Pipeline de ingestão de documentos
- **[src/chain.py](../src/chain.py)** - Configuração da chain RAG
- **[src/query.py](../src/query.py)** - Interface de consultas

### Exemplos
- **[examples/](../examples/)** - Exemplos práticos de uso

### Scripts
- **[scripts/](../scripts/)** - Scripts auxiliares e utilitários

## 🤝 Contribuindo

Quer contribuir para a documentação? Veja as diretrizes:

1. **Clareza**: Use linguagem clara e exemplos práticos
2. **Estrutura**: Siga o padrão Markdown dos arquivos existentes
3. **Links**: Use links relativos para outros documentos
4. **Exemplos**: Inclua exemplos de código quando relevante
5. **Atualização**: Mantenha o Changelog atualizado

### Como adicionar nova documentação:

1. Crie o arquivo `.md` em `docs/` ou `docs/guides/`
2. Adicione link neste README.md
3. Use cabeçalhos e índice para navegação
4. Teste todos os links
5. Atualize o Changelog

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/patrickmcruz/rag-demo/issues)
- **Discussões**: [GitHub Discussions](https://github.com/patrickmcruz/rag-demo/discussions)
- **Email**: patrickmcruz@gmail.com

## 📄 Licença

Este projeto está sob a licença GNU General Public License. Veja [LICENSE](../LICENSE) para mais detalhes.

---

**Última atualização**: Novembro 2025  
**Versão da documentação**: 1.0.0
