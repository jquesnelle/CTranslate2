#include <sentencepiece_processor.h>
#include <ctranslate2/generator.h>
#include <ctranslate2/generation.h>
#include <ctranslate2/models/language_model.h>
#include <iostream>
#include <regex>
#include <chrono>

static std::vector<std::string_view> get_vocabulary_tokens(const ctranslate2::Vocabulary& vocabulary) {
  std::vector<std::string_view> tokens;
  const size_t size = vocabulary.size();
  tokens.reserve(size);
  for (size_t i = 0; i < size; ++i)
    tokens.emplace_back(vocabulary.to_token(i));
  return tokens;
}

int main(int, char* argv[]) {
  const std::string model_path = argv[1];
  const std::string sp_model_path = model_path + "/tokenizer.model";

  const ctranslate2::Device device = ctranslate2::str_to_device("auto");
  ctranslate2::ComputeType compute_type = ctranslate2::ComputeType::INT8;

  const auto model = ctranslate2::models::Model::load(model_path, device, 0, compute_type);
  ctranslate2::Generator generator(model);

  sentencepiece::SentencePieceProcessor sp_processor;
  auto status = sp_processor.Load(sp_model_path);
  if (!status.ok())
    throw std::invalid_argument("Unable to open SentencePiece model " + sp_model_path);
  const auto* language_model = dynamic_cast<const ctranslate2::models::LanguageModel*>(model.get());
  status = sp_processor.SetVocabulary(get_vocabulary_tokens(language_model->get_vocabulary()));
  if (!status.ok())
    throw std::runtime_error("Failed to set the SentencePiece vocabulary");

  auto tokenizer = [&sp_processor](const std::string& text) {
    std::vector<std::string> tokens;
    sp_processor.Encode(text, &tokens);
    return tokens;
  };

  auto detokenizer = [&sp_processor](const std::vector<std::string>& tokens) {
    std::string text;
    sp_processor.Decode(tokens, &text);
    return std::regex_replace(text, std::regex("<unk>"), "UNK");
  };

  ctranslate2::GenerationOptions options;
  options.sampling_temperature = 0.7;
  options.sampling_topk = 40;
  options.max_length = 512;

  std::vector<int> output_ids;
  options.callback = [&](ctranslate2::GenerationStepResult& result) {
    const bool is_new_word = result.token.size() >= 3 && result.token[0] == -30 && result.token[1] == -106 && result.token[2] == -127;

    if (is_new_word && !output_ids.empty()) {
        std::string word;
        sp_processor.Decode(output_ids, &word);
        std::cout << word << " ";
        output_ids.clear();
    }
  
    output_ids.push_back((int)result.token_id);
    return false;
  };

  std::vector<std::vector<std::string>> prompt_tokens;
  prompt_tokens.push_back(tokenizer(argv[2]));

  auto start = std::chrono::high_resolution_clock::now();
  auto promise = generator.generate_batch_async(prompt_tokens, options);
  auto result_tokens = promise[0].get();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  auto ms_per_token = (float)duration.count() / result_tokens.sequences_ids[0].size();

  std::cout << std::endl << std::endl << "ms per token: " << ms_per_token << std::endl;

  return 0;
}
