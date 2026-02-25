export type Metadata = {
    total_pages: number;
    title?: string;
    author?: string;
    creation_date?: string;
};

export type TextRow = {
    file_name: string;
    page_num: number;
    text: string;
    chunk_number: number;
    chunk_text: string;
    text_embedding_chunk: number[];
};

export type ImageRow = {
    file_name: string;
    page_num: number;
    img_num: number;
    img_path: string;
    img_desc: string;
    mm_embedding_from_img_only: number[];
    text_embedding_from_image_description: number[];
};
