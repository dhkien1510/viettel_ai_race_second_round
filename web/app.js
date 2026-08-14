const form = document.querySelector('#search-form');
const input = document.querySelector('#query');
const results = document.querySelector('#results');
const status = document.querySelector('#status');
const title = document.querySelector('#result-title');
const count = document.querySelector('#result-count');
const template = document.querySelector('#result-template');
const ttyLegend = document.querySelector('#tty-legend');

const TTY_LABELS = {
  IN: ['Hoạt chất', 'Tên hoạt chất ở mức chung, chưa có hàm lượng hoặc dạng dùng.'],
  PIN: ['Hoạt chất chính xác', 'Dạng hoạt chất cụ thể, chẳng hạn muối hoặc ester.'],
  MIN: ['Nhiều hoạt chất', 'Khái niệm gồm từ hai hoạt chất trở lên.'],
  BN: ['Tên thương mại', 'Tên nhãn hiệu, chưa nhất thiết chỉ rõ hàm lượng và dạng dùng.'],
  SCD: ['Thuốc lâm sàng', 'Hoạt chất + hàm lượng + dạng dùng, không gắn thương hiệu.'],
  SBD: ['Thuốc biệt dược', 'Sản phẩm có hàm lượng, dạng dùng và thương hiệu.'],
  SCDC: ['Thành phần thuốc lâm sàng', 'Hoạt chất + hàm lượng, chưa chỉ rõ dạng dùng.'],
  SBDC: ['Thành phần biệt dược', 'Thành phần có hàm lượng thuộc một biệt dược.'],
  SCDF: ['Dạng thuốc lâm sàng', 'Hoạt chất + dạng dùng, chưa chỉ rõ hàm lượng.'],
  SBDF: ['Dạng biệt dược', 'Thương hiệu + dạng dùng, chưa chỉ rõ hàm lượng.'],
  PSN: ['Tên kê đơn', 'Tên được RxNorm trình bày theo cách phù hợp để kê đơn.'],
  SY: ['Tên đồng nghĩa', 'Cách gọi khác của cùng khái niệm RxNorm.'],
  TMSY: ['Tên Tall Man', 'Tên đồng nghĩa dùng viết hoa chọn lọc để giảm nhầm lẫn thuốc.'],
  SCDG: ['Nhóm thuốc lâm sàng', 'Nhóm các thuốc lâm sàng có đặc điểm chung.'],
  SBDG: ['Nhóm biệt dược', 'Nhóm các thuốc biệt dược có đặc điểm chung.'],
  SCDGP: ['Gói thuốc lâm sàng', 'Nhóm/gói chứa nhiều thuốc lâm sàng.'],
  SCDFP: ['Gói dạng thuốc lâm sàng', 'Gói thuốc được mô tả ở mức dạng dùng.'],
  SBDFP: ['Gói dạng biệt dược', 'Gói biệt dược được mô tả ở mức dạng dùng.'],
};

function ttyText(code) {
  const info = TTY_LABELS[code];
  return info ? `${code} — ${info[0]}` : code || 'Không xác định';
}

Object.entries(TTY_LABELS).forEach(([code, [label, description]]) => {
  const item = document.createElement('div');
  item.innerHTML = `<strong>${code}</strong><span>${label}</span><p>${description}</p>`;
  ttyLegend.append(item);
});

let activeRequest = null;

function setLoading(query) {
  results.replaceChildren();
  title.textContent = `Đang tìm “${query}”`;
  status.textContent = 'Đang truy vấn cơ sở dữ liệu RxNorm…';
  status.hidden = false;
  count.hidden = true;
}

function render(data) {
  results.replaceChildren();
  title.textContent = data.query ? `Kết quả cho “${data.query}”` : 'Nhập từ khóa để bắt đầu';
  count.textContent = `${data.count} kết quả`;
  count.hidden = false;

  if (!data.results.length) {
    status.textContent = 'Không tìm thấy thuốc phù hợp. Hãy thử tên hoạt chất, biệt dược khác hoặc RxCUI.';
    status.hidden = false;
    return;
  }
  status.hidden = true;

  data.results.forEach((drug, index) => {
    const node = template.content.cloneNode(true);
    const card = node.querySelector('.drug-card');
    card.style.animationDelay = `${Math.min(index * 25, 180)}ms`;
    node.querySelector('h3').textContent = drug.name;
    node.querySelector('.match').textContent = drug.match;
    node.querySelector('.rxcui').textContent = drug.rxcui;
    const tty = node.querySelector('.tty');
    tty.textContent = ttyText(drug.tty);
    tty.title = TTY_LABELS[drug.tty]?.[1] || '';
    node.querySelector('.state').textContent = drug.suppressed ? 'Suppressed' : 'Đang hoạt động';

    const list = node.querySelector('.synonyms');
    drug.names.forEach(item => {
      const li = document.createElement('li');
      const name = document.createElement('span');
      const tty = document.createElement('small');
      name.textContent = item.name;
      tty.textContent = ttyText(item.tty);
      tty.title = TTY_LABELS[item.tty]?.[1] || '';
      li.append(name, tty);
      list.append(li);
    });

    const copy = node.querySelector('.copy');
    copy.addEventListener('click', async () => {
      await navigator.clipboard.writeText(drug.rxcui);
      copy.textContent = 'Đã chép';
      setTimeout(() => { copy.textContent = 'Sao chép'; }, 1200);
    });
    results.append(node);
  });
}

async function search(query) {
  query = query.trim();
  if (!query) {
    render({ query: '', count: 0, results: [] });
    return;
  }
  activeRequest?.abort();
  activeRequest = new AbortController();
  setLoading(query);
  try {
    const response = await fetch(`api/search?q=${encodeURIComponent(query)}&limit=20`, {
      signal: activeRequest.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    if (error.name === 'AbortError') return;
    status.hidden = false;
    status.classList.add('error');
    status.textContent = 'Không thể kết nối tới database. Hãy kiểm tra server Python.';
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  search(input.value);
});

document.querySelectorAll('[data-query]').forEach(button => {
  button.addEventListener('click', () => {
    input.value = button.dataset.query;
    search(input.value);
  });
});
