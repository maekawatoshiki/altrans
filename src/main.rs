use std::io::Write;

use altius_core::{onnx::load_onnx, tensor::Tensor};
use altius_session::interpreter::InterpreterSessionBuilder;
use dirs;
use tokenizers::Tokenizer;

fn main() {
    let cache_dir = dirs::home_dir().unwrap().join(".cache/altrans");
    let tokenizer = Tokenizer::from_file(cache_dir.join("fugumt/target-tokenizer.json")).unwrap();

    let encoder = load_onnx(cache_dir.join("fugumt/encoder_model.onnx")).unwrap();
    let encoder_input_ids = encoder.lookup_named_value("input_ids").unwrap();
    let encoder_attention_mask = encoder.lookup_named_value("attention_mask").unwrap();
    let encoder = InterpreterSessionBuilder::new(&encoder)
        .with_intra_op_num_threads(8)
        .build()
        .unwrap();

    let decoder = load_onnx(cache_dir.join("fugumt/decoder_model.onnx")).unwrap();
    let decoder_encoder_attention_mask = decoder
        .lookup_named_value("encoder_attention_mask")
        .unwrap();
    let decoder_input_ids = decoder.lookup_named_value("input_ids").unwrap();
    let decoder_encoder_hidden_states =
        decoder.lookup_named_value("encoder_hidden_states").unwrap();
    let decoder = InterpreterSessionBuilder::new(&decoder)
        .with_intra_op_num_threads(8)
        .build()
        .unwrap();

    loop {
        let mut text = "".to_string();
        std::io::stdin().read_line(&mut text).unwrap();
        if text.is_empty() {
            break;
        }
        let encoding = tokenizer.encode(text, false).unwrap();

        let encoder_attention_mask_tensor = {
            let mut attn_mask = vec![0i64; 100];
            for i in 0..encoding.get_ids().len() + /*</s> = */1 {
                attn_mask[i] = 1;
            }
            Tensor::new(vec![1, 100].into(), attn_mask)
        };

        let last_hidden_state = encoder
            .run(vec![
                (encoder_input_ids, {
                    let mut input_ids = vec![0u32; 100];
                    let ids = encoding.get_ids();
                    input_ids[0..ids.len()].copy_from_slice(ids);
                    Tensor::new(
                        vec![1, 100].into(),
                        input_ids.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
                    )
                }),
                (
                    encoder_attention_mask,
                    encoder_attention_mask_tensor.clone(),
                ),
            ])
            .unwrap()
            .remove(0);
        // println!("{}", last_hidden_state);

        let mut decoder_input_ids_vec = vec![32000i64];

        for idx in 0..100 {
            let logits = decoder
                .run(vec![
                    (
                        decoder_encoder_attention_mask,
                        encoder_attention_mask_tensor.clone(),
                    ),
                    (decoder_input_ids, {
                        let mut input_ids = vec![0i64; 100];
                        input_ids[0..decoder_input_ids_vec.len()]
                            .copy_from_slice(decoder_input_ids_vec.as_slice());
                        Tensor::new(vec![1, 100].into(), input_ids)
                    }),
                    (decoder_encoder_hidden_states, last_hidden_state.clone()),
                ])
                .unwrap()
                .remove(0);
            let mut indices = (0..logits.dims()[2]).collect::<Vec<_>>();
            let logits = logits.slice_at::<f32>(&[0, idx]);
            indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let mut next_token = 0i64;

            for i in indices {
                if i == 32000 {
                    // pad
                    continue;
                }
                // if i == 2 {
                //     // space
                //     continue;
                // }
                next_token = i as i64;
                break;
            }
            if next_token == 0 {
                // eos
                break;
            }
            decoder_input_ids_vec.push(next_token);
            let translation = tokenizer
                .decode(
                    decoder_input_ids_vec
                        .clone()
                        .into_iter()
                        .map(|x| x as u32)
                        .collect(),
                    true,
                )
                .unwrap();
            print!("{}\r", translation);
            std::io::stdout().flush().unwrap();
        }
        println!()
    }
}
